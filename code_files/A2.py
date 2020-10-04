# WARNING: Runs with Python 3.7.5. on the exported environment.
import numpy
# Using SoundFile library in order to read and write sound(wav) files.
import soundfile
# The functions that we implemented for the program
from MyFunctions import MyConvolve

#Reads the signal and sample rate of the audio file and pink noise from the .wav files.
sampleAudioData, sampleAudioSampleRate = soundfile.read('sample_audio.wav')
pinkNoiseData, pinkNoiseSampleRate = soundfile.read('pink_noise.wav')

# Calculates and writes in a .wave file the output of the convolution between the audio file and the pink noise.
pinkNoise_sampleAudioData = MyConvolve(pinkNoiseData, sampleAudioData)
#print("pinkNoise_sampleAudioData size: ", len(pinkNoise_sampleAudioData)) #Size of the output
#print("pinkNoise_sampleAudioData: ", pinkNoise_sampleAudioData) # The convolution signal
soundfile.write('pinkNoise_sampleAudio.wav', pinkNoise_sampleAudioData, sampleAudioSampleRate)
print("Done with the sampleAudio*PinkNoise Convolution.")

# White noise is a signal whose samples are regarded as a sequence of serially uncorrelated random variables with zero mean and finite variance.
# if each sample has a normal distribution with zero mean, the signal is said to be additive white Gaussian noise.
# In order to make this possible we use numpy.random.normal(), which draws a given number of samples from a Gaussian distribution.
mean = 0
std = 1
samples = len(pinkNoiseData)
whiteNoiseData = numpy.random.normal(mean, std, samples)
soundfile.write('white_noise.wav', whiteNoiseData, pinkNoiseSampleRate)
print("Done writing the WhiteNoise file.")

# Calculates and writes in a .wave file the output of the convolution between the audio file and the white noise.
whiteNoise_sampleAudioData = MyConvolve(whiteNoiseData, sampleAudioData)
#print("whiteNoise_sampleAudioData size: ", len(whiteNoise_sampleAudioData)) #Size of the output
#print("whiteNoise_sampleAudioData: ", whiteNoise_sampleAudioData) # The convolution signal
soundfile.write('whiteNoise_sampleAudio.wav', whiteNoise_sampleAudioData, sampleAudioSampleRate)
print("Done with the sampleAudio*WhiteNoise Convolution.")