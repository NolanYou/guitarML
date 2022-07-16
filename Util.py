from pydub import AudioSegment
import math


class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename

        self.audio = AudioSegment.from_wav(self.filepath)

    def get_duration(self):
        return self.audio.duration_seconds

    def single_split(self, from_ms, to_ms, split_filename):
        t1 = from_ms
        t2 = to_ms
        split_audio = self.audio[t1:t2]
        split_audio.export(self.folder + '/' + split_filename, format="wav")

    def multiple_split(self, secs_per_split):
        total_ms = self.get_duration() * 1000
        ms_per_split = secs_per_split * 1000
        for i in range(0, int(total_ms), ms_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i + ms_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_ms - ms_per_split:
                print('All splited successfully')
if __name__ == "__main__":
    out =  SplitWavAudioMubin("/Users/nolan/PycharmProjects/guitarML", "MachineLearningAudio - 7:15:22, 3.57 PM.wav")
    out.multiple_split(3)