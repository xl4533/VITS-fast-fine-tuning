import time
import os
import random
import numpy as np
import torch
import torch.utils.data
import torchaudio

import commons
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence
"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """

    def __init__(self, audiopaths_sid_text, hparams, symbols):
        #加载音频数据的路径和对应的文本
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)#加载音频文件名和文本
        #定义音频和文本处理所需的参数
        self.text_cleaners = hparams.text_cleaners#加载文本清理器
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate#音频采样率
        self.filter_length = hparams.filter_length#过滤器长度
        self.hop_length = hparams.hop_length#帧移
        self.win_length = hparams.win_length#窗口长度
        self.sampling_rate = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)#getattr(object, name[, default])其中，object是要获取属性值或方法的对象；name是要获取的属性名或方法名；default是可选参数，当指定的属性或方法不存在时，会返回default的值1。
        self.add_blank = hparams.add_blank#是否在文本的token序列间增加空格，论文中提到可提高模型效果
        self.min_text_len = getattr(hparams, "min_text_len", 1)#文本最小长度
        self.max_text_len = getattr(hparams, "max_text_len", 190)#文本最大长度
        self.symbols = symbols

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()#数据过滤和存储是spec长度

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing，方便后续进行桶排序
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            # audiopath = "./user_voice/" + audiopath

            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:#如果文本长度符合要求
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))#计算spec长度并记录
        self.audiopaths_sid_text = audiopaths_sid_text_new#将过滤后的数据重新复制
        self.lengths = lengths#所有spec序列的长度

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2]
        text = self.get_text(text)#加载经过处理后的文本序列
        spec, wav = self.get_audio(audiopath)#加载对应的spec序列和音频数据
        sid = self.get_sid(sid)#将sid转为张量
        return (text, spec, wav, sid)#返回处理后的文本呢数值序列，spec序列，音频数据以及speaker的id

    def get_audio(self, filename):
        # audio, sampling_rate = load_wav_to_torch(filename)
        # if sampling_rate != self.sampling_rate:
        #     raise ValueError("{} {} SR doesn't match target {} SR".format(
        #         sampling_rate, self.sampling_rate))
        # audio_norm = audio / self.max_wav_value if audio.max() > 10 else audio
        # audio_norm = audio_norm.unsqueeze(0)
        audio_norm, sampling_rate = torchaudio.load(filename, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
        # spec_filename = filename.replace(".wav", ".spec.pt")
        # if os.path.exists(spec_filename):
        #     spec = torch.load(spec_filename)
        # else:
        #     try:
        spec = spectrogram_torch(audio_norm, self.filter_length,
                                 self.sampling_rate, self.hop_length, self.win_length,
                                 center=False)#从标准化后的音频数据中提取spec谱图
        spec = spec.squeeze(0)
            # except NotImplementedError:
            #     print("?")
            # spec = torch.squeeze(spec, 0)
            # torch.save(spec, spec_filename)
        return spec, audio_norm

    def get_text(self, text):
        if self.cleaned_text:#是否是已经经过clean过的文本
            text_norm = cleaned_text_to_sequence(text, self.symbols)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)#在所有文本数值序列中元素的前后都补充一个0
        text_norm = torch.LongTensor(text_norm)#转为long的张量
        return text_norm

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)

#基于dataset构建dataloader时调用的回调函数，主要是对各类序列进行pad
class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)#基于spec序列的长度进行降序排序，返回的是对应的索引列表
        #计算batch中各类数据序列的最大长度
        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        #为各类序列构建存储序列长度的容器，长度都是batch_size
        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        #先以最大长度构建尺寸正确的数据背景，方便后续数据填充，没有填充位置就相当于用0进行了pad
        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        #全部先用0进行初始化
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

        if self.return_ids:#是否返回排序后的索引
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid

#的是因为不同音频之间长度变化可能很大&#xff0c;先通过同排序将长度差别较少的放入同一桶中;batch从一个桶中取数据;减少同一batch中序列长度的差别。可提高训练效率
class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.#维持一个batch中输入长度相似
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.#不在长度设置的boundaries范围内的音频数据被丢弃
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths#记录所有数据中spec序列的长度
        self.batch_size = batch_size#batch大小
        self.boundaries = boundaries#定义不同桶之间的边界，如0~30放在一个桶里，30~60放在一个桶里，单位是帧

        self.buckets, self.num_samples_per_bucket = self._create_buckets()#定义桶，将dataset中的数据基于长度分配给对应的桶
        self.total_size = sum(self.num_samples_per_bucket)#将记录所有桶“应该”包含的数据个数全部相加
        self.num_samples = self.total_size // self.num_replicas#每张卡，在每个epoch会见到的数据数量

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]#初始将每个桶定义为一个空列表
        for i in range(len(self.lengths)):#遍历dataset中每条音频序列的长度
            length = self.lengths[i]#获取当前音频序列对应的长度
            idx_bucket = self._bisect(length)#基于length进行桶排序，返回其应该属于的桶的下标索引
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)#将该音频文件在dataset中的索引存放至对应桶中

        try: 
            for i in range(len(buckets) - 1, 0, -1):#从后往前遍历桶
                if len(buckets[i]) == 0:#如果该桶中没有存放到数据
                    buckets.pop(i)#删除桶
                    self.boundaries.pop(i + 1)#删除最高的上界
            assert all(len(bucket) > 0 for bucket in buckets)
        # When one bucket is not traversed
        except Exception as e:
            print('Bucket warning ', e)
            for i in range(len(buckets) - 1, -1, -1):
                if len(buckets[i]) == 0:
                    buckets.pop(i)
                    self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])#每个桶中数据个数
            total_batch_size = self.num_replicas * self.batch_size#卡数乘上每个卡上的batch_size就是total_batch_size。注：桶的真实个数不一定是total_batch_size的整数倍，rem离整数倍还差的值
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)#加上rem，保证每个桶的个数都是rem的整数倍
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)#将epoch设置为生成器g的随机种子

        indices = []#存储每个桶中数据的索引
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())#会将桶中数据的索引进行打乱
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []#存储所有minibatch的数据；遍历完所有的桶，就是一个卡在一个epoch被分配的所有数据
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]#获取一个桶
            len_bucket = len(bucket)#bucket中当前存放数据索引的个数，可能还不是total_batch_size的整数倍
            ids_bucket = indices[i]#获取该bucket中数据的索引
            num_samples_bucket = self.num_samples_per_bucket[i]#number_sample_bucket是进行了余数补偿，是total_batch_size的整数倍

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket#计算bucket中数据量是batch_size的整数倍所差的数据量
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]#从ids_bucket中抽取一些索引，数据量是rem，补全batch中的数据，使得数据量是total_batch_size的整数倍

            # subsample，ids_bucket中存放当前桶分配给卡rank的所有数据
            ids_bucket = ids_bucket[self.rank::self.num_replicas]#给每个gpu分配数据，rank表示当前是第几张卡，num_replicas是总卡数

            # batching，一个卡在一个桶中被分配的所有batch的数据
            for j in range(len(ids_bucket) // self.batch_size):#j遍历的是batch的个数
                batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]#一个batch的数据
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()#对batch进行shuffle
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples#len(self.batches)是所有的batch的数量，乘上batch_size，就是该卡一个epoch中所有的数据量，与前面的self.num_samples要相等
        return iter(self.batches)
    #基于二分法的桶排序
    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
