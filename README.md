# Papers-Reading
My reading notes on DL papers, along with my personal comment of each paper, so there may exist lots of mistakes, I really appreciate you to point out.

## Neural Style Transfer
- [x] [Neural Style Transfer: A Review](https://github.com/fancoo/Papers-Reading/blob/master/Neural-Style-Transfer/Neural%20Style%20Transfer-A%20Review.pdf) :star::star::star::star:
	* Investigate the works of Neural Style Transfer till May of 2016.
- [ ] [Demystifying Neural Style Transfer](https://arxiv.org/abs/1701.01036)
	* Prove that matching the Gram matrices is actually equivalent to minimize the Maximum Mean Discrepancy(MMD) with second order polynomial kernel.
	* Try out for different kernels and parameters.
- [ ] [Fast Patch-based Style Transfer of Arbitrary Style](https://arxiv.org/abs/1612.04337)
	* A more advanced version of "Fast" Neural Style Transfer that can run in real-time and applies to infinite kind of styles.
	* The drawback is the quality of stylized images is worse than "Fast" Neural Style which yet can only applies to finite styles.

## Generative Model
- [x] [Pixel Recurrent Neural Networks(Best Paper of ICML2016)](https://github.com/fancoo/Papers-Reading/blob/master/Generative-Model/Pixel%20Recurrent%20Neural%20Networks.pdf) :star::star::star::star:
	* I quickly skimmed this paper, it introduced a new method to generate image pixel by pixel with sequence model, which means **you can only predict current pixel by it's previous pixels(namely the pixels above and to the left of it).** To achieve this, they introduce a `mask` to make sure model can not read later pixels.
	* The loss curve is much more smooth and interpretatable compared to GAN.
- [x] [Conditional Image Generation with PixelCNN Decoders](https://github.com/fancoo/Papers-Reading/blob/master/Generative-Model/Conditional%20Image%20Generation%20with%20PixelCNN%20Decoders.pdf) :star::star::star::star::star:
	* An improvement to PixelRNN & PixelCNN by adding an additional `Gated activation unit`.
	* Use two stack(vertical and horizontal) to aviod the `blind spot` in Mask.
	* Explore the performance of image generation in this kind of `Gated PixelCNN` in conditional distribution image, actually it seems not as good as GAN but, still another method and therefore lead to the famous [WaveNet](https://arxiv.org/abs/1609.03499).
- [x] [WaveNet: A Generative Model for Raw Audio](https://github.com/fancoo/Papers-Reading/blob/master/Generative-Model/WaveNet_%20A%20Generative%20Model%20for%20Raw%20Audio.pdf) :star::star::star::star::star:
	* A summary of papers of above, and use these methods in audio.
	* Keywords: fuse the technic of `Dilated Casual Convolution`, `Gated Activation Units` and `residual network` along with `skip connections`.
	* Based on Conditional WaveNet, they explored the experiments of `Multi-Speaker Speech Generation`, `TTS(Text-To-Speech)` and `Music Generation` by feeding additional input `h`. In speech generation, it's speaker ID of one-hot vector, in TTS it's the text while in music generation it's the tag of generated musich, like the instruments or the genre.
- [ ] [Parallel WaveNet: Fast High-Fidelity Speech Synthesis](https://arxiv.org/abs/1711.10433)

### GAN for Image
- [x] [Self-Attention Generative Adversarial Networks(!!Important)](https://github.com/fancoo/Papers-Reading/blob/master/Generative-Model/Self-Attention%20Generative%20Adversarial%20Networks.pdf):star::star::star::star::star:
	* Self-Attention GAN, boosting the best published Inception score from **36.8** to **52.52** and reducing Frechet Inception distance from **27.62** to **18.65** on the challenging ImageNet dataset.
	* Using Self-Attention to learn long-range dependency. 
	* Several tricks inside:
		* Used Spectral-Normalization both on generator and discriminater, it proved to be more stable when training compared with `SN-GAN`.
		* Showed two-timescale update rule (TTUR) is an effect way for faster converge.
		* Indicated that self-attention mechanism at the middle-to-high level feature maps (e.g., feat32 and feat64) achieve better performance than at low level feature maps. The reason could be that the network receives more evidence with larger feature maps and enjoys more freedom to choose the conditions.
- [x] [Conditional Generative Adversarial Nets](https://github.com/fancoo/Papers-Reading/blob/master/Generative-Model/Conditional%20Generative%20Adversarial%20Nets.pdf):star::star::star::star:
	* cGAN, you can embed information to control the generated result.
	* The information is feeded both into generator & discriminator. This can be done by concating the `z`(after fc) with label `y`(after fc).  
	* They experimented on MNIST generation with given number as `y`(one-hot), and a multimodel tagging, especially for the tagging work, they use an image as information by letting it pass through pretrained CNN to be the `y`.


- [ ] [Wasserstein GAN](https://arxiv.org/abs/1701.07875)

### Attention
- [ ] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### GAN for Text&Audio generation
- [ ] [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient]()
- [x] [Synthesizing Audio with Generative Adversarial Networks](https://github.com/fancoo/Papers-Reading/blob/master/Generative-Model/Synthesizing%20Audio%20with%20Generative%20Adversarial%20Networks.pdf):star::star::star::star:
	* The first listenable GAN based audio generation work.
	* Using several methods as below:
		* 1D Conv(filter len=25) rather thant 5x5
		* Upsample by factory of 4 at each layer
		* Learned post processing filter & **Phase shuffle** to prevent discriminator learning to classify fake/real audio only by phase.
	* Explore WaveGAN and SpecGAN, though the Inception Score of SpecGAN is higher (6.0) than WaveGAN(4.7), human prefer more about WaveGAN.(So is this means IC criterion can be updated ? Or means SpecGAN has some potential ?)
	* Give a 0-9 audio dataset SC09.
- [x] [C-RNN-GAN: Continuous recurrent neural networks with adversarial training](https://github.com/fancoo/Papers-Reading/blob/master/Generative-Model/C-RNN-GAN_%20Continuous%20recurrent%20neural%20networks%20with%20adversarial%20training.pdf):star::star::star::star:
	* LSTM based Generater and Discriminator with the dataset of MIDI classic work.
	* Apply trick such as `curriculum learning`(continuing increase sequence length.), `freezing`(control the capability of G and D) and `feature matching`(I don't understand here...)
	* **Evaluation**: `Polyphony`, `Scale consistency`, `Repetitions`, `Tone span`.
- [ ] [Semi-Recurrent CNN-based VAE-GAN for Sequential Data Generation(ICASSP 2018)]()
- [ ] [A Note on the Inception Score(ICML 2018 Workshop)]()
- [ ] [MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment(AAAI 2018)]()
- [ ] [MidiNet: A Convolutional Generative Adversarial Network for Symbolic-domain Music Generation(ISMIR’17)]()
- [ ] [Language Generation with Recurrent Generative Adversarial Networks without Pre-training(ICML 2017 Workshop)]()

## Speech
### WaveNet(Mentioned before)

### Tactron
- [ ] [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
- [ ] [Tacotron series](https://google.github.io/tacotron/index.html)

### Deep Voice
- [ ] [Deep Voice: Real-time Neural Text-to-Speech](https://arxiv.org/abs/1702.07825)
- [ ] [Deep Voice 2: Multi-Speaker Neural Text-to-Speech](https://arxiv.org/abs/1705.08947)
- [ ] [Deep Voice 3: Scaling Text-to-Speech with Convolutional Sequence Learning](https://arxiv.org/abs/1710.07654)
- [ ] [Neural Voice Cloning with a Few Samples](https://arxiv.org/abs/1802.06006)
	* A fresh new paper by `Baidu` of using a few samples to generate a lot of TTS audio.

### Others
- [x] [Towards End-to-End Speech Recognition with Deep Convolutional Neural](https://github.com/fancoo/Papers-Reading/blob/master/Speech/Towards%20End-to-End%20Speech%20Recognition%20with%20Deep%20Convolutional%20Neural.pdf) :star::star::star:
	* They found it's possiable to use **only CNN based** end-to-end model to do *Speech recognition*(**SR**) task, the results is as good as those of RNNs.
	* They treat audio spectrogram as 2-D CNN, building with `CONV2D + Maxout + CTC` archicture and finally evaluating the model in TIMIT dataset.

## Voice Transfer
Papers related with my current research.
- [x] [Singing Expression Transfer from One Voice to Another for a Given Song](https://github.com/fancoo/Papers-Reading/blob/master/Voice-Transfer/SINGING%20EXPRESSION%20TRANSFER%20FROM%20ONE%20VOICE%20TO%20ANOTHER%20FOR%20A%20GIVEN%20SONG.pdf):star::star::star:
	* I skim the paper, it introduced a method of to improve our singing records: first we have a `source` audio(my voice), and a `target` audio which we want to sing as well as him/her. We first align the two pieces voice and compare them frame-by-fram with some features like phoneme etc.
	* Their sample is [Singing Expression Transfer](https://seyong92.github.io/singing-expression-transfer/), the results are not so good and it's apparently not what I'm interested in.
- [x] [Time Domain Neural Audio Style Transfer(NIPS2017)](https://github.com/fancoo/Papers-Reading/blob/master/Voice-Transfer/Time%20Domain%20Neural%20Audio%20Style%20Transfer.pdf):star::star::star:
	* This paper presents a method for audio style transfer by directly optimizing a time domain audio signal, it explores many architectures(e.g WaveNet encoder/NSynth encoder), the result is almost the same as `Ulyanov` but real-time?
	* The github implementation is [time-domain-neural-audio-style-transfer](https://github.com/pkmital/time-domain-neural-audio-style-transfer)
- [ ] [Universal Style Transfer via Feature Transforms](https://arxiv.org/abs/1705.08086)
- [x] [Audio Style Transfer](https://github.com/fancoo/Papers-Reading/blob/master/Voice-Transfer/Audio%20Style%20Transfer.pdf):star::star::star:
	* Try with `VGG-19`, `SoundNet`, `Wide-Shallow-Random Network` and `McDermott's texture synthesis` method to extract the style. The last 2 has meaningful results, `McDermott's..` recreates better local texture.
	* Shows that **starting from content image can produce better results** comapred to random generated.
- [x] [On Using Backpropagation for Speech Texture Generation and Voice Conversion(Google, 2018.03)](https://github.com/fancoo/Papers-Reading/blob/master/Voice-Transfer/%20On%20Using%20Backpropagation%20for%20Speech%20Texture%20Generation%20and%20Voice%20Conversion.pdf):star::star::star::star:
	* Use the architecture of CTC speech recognition network to train a CONV.
	* Extract speaker characteristics from very limited amounts of target speaker data.
	* Did the experiment of `style transfer` and others, the result page is [here](https://google.github.io/speech_style_transfer/samples.html)
- [x] [Audio spectrogram representations for processing with Convolutional Neural Networks](https://github.com/fancoo/Papers-Reading/blob/master/Voice-Transfer/Audio%20spectrogram%20representations%20for%20processing%20with%20Convolutional%20Neural%20Networks.pdf):star::star::star:
	* The major contribution of this paper is, to go on [Ulyanov's idea](https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/) by training a network with two convolutional layers and two
fully-connected layers on the [ESC-50](https://github.com/karoldvl/ESC-50) data, and replaced the original `random param CNN` with pre-trained CNN.
	* Their [result demo](http://lonce.org/research/audioST/) shows that **although style transfer does work without regard to weights, a network trained for
audio classification appears to generate a more integrated synthesis of content and style**
- [ ] [ObamaNet: Photo-realistic lip-sync from text](https://arxiv.org/pdf/1801.01442.pdf)
- [ ] [One-shot learning of generative speech concepts](https://cims.nyu.edu/~brenden/LakeLeeEtAl2014CogSci.pdf)
- [ ] [CHAR2WAV: END-TO-END SPEECH SYNTHESIS](https://mila.quebec/wp-content/uploads/2017/02/end-end-speech.pdf)
- [ ] [The Voice Conversion Challenge 2016](http://www.vc-challenge.org/vcc2016/summary.html)
- [ ] [A Fully Convolutional Neural Network for Speech Enhancement](https://arxiv.org/abs/1609.07132)
	* A paper described how to use CNN for removing babble noise in a audio so as to enchance our hearing. It used encoder-decoder which may worthy to read.

### Some most related work !
- [ ] [“Style” Transfer for Musical Audio Using Multiple Time-Frequency Representations(ICLR 2018 Rejected)](https://openreview.net/forum?id=BybQ7zWCb)
	* Github: [Style-Transfer-for-Musical-Audio](https://github.com/anonymousiclr2018/Style-Transfer-for-Musical-Audio)
- [ ] [Time Domain Neural Audio Style Transfer(NIPS 2017 Workshop)](https://arxiv.org/pdf/1711.11160.pdf)
	* Github: [time-domain-neural-audio-style-transfer](https://github.com/pkmital/time-domain-neural-audio-style-transfer)
- [x] [On Using Backpropagation for Speech Texture Generation and Voice Conversion(Google, 2018.03)](https://github.com/fancoo/Papers-Reading/blob/master/Voice-Transfer/%20On%20Using%20Backpropagation%20for%20Speech%20Texture%20Generation%20and%20Voice%20Conversion.pdf):star::star::star::star:
- [ ] [Neural Style Transfer for Audio Spectrograms(NIPS 2017 Workshop)](https://arxiv.org/abs/1801.01589)
- [ ] [Audio texture synthesis and style transfer(Blog)](https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer/)
	* Github: [neural-style-audio-tf](https://github.com/DmitryUlyanov/neural-style-audio-tf)

### Random CNN Based
- [x] [A Powerful Generative Model Using Random Weights for the Deep Image Representation(NIPS 2016)](https://github.com/fancoo/Papers-Reading/blob/master/Voice-Transfer/A%20Powerful%20Generative%20Model%20Using%20Random%20Weights%20for%20the%20Deep%20Image%20Representation.pdf) :star::star::star::star:
	* This paper shows untrained network can be used for image representation. It used random weights for VGG archicture to do `Inverting deep representation`, `Texture synthesis` and `Style transfer`. And the result is comparable with the pretrained VGG.
	* It shows we can use this for archicture comparison without training them, so we can save a lot of time of comparing different archictures. 
- [ ] [Texture Synthesis Using Shallow Convolutional Networks with Random Filters](https://arxiv.org/abs/1606.00021)
- [ ] [Extreme Style Machines:
Using Random Neural Networks to Generate Textures](https://nucl.ai/blog/extreme-style-machines/)
- [ ] [On Random Weights and
Unsupervised Feature Learning(ICML 2011)](http://www.robotics.stanford.edu/~ang/papers/nipsdlufl10-RandomWeights.pdf)

### Self-Attention Based
- [ ] [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
	* The first paper of **Self-Attention** proposed by Google.
- [ ] [Github:Deep-Expression](https://github.com/ttsunion/Deep-Expression)
	* A github repo using only Self-Attention for TTS.
- [ ] [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)
	* Han Zhang, Ian Goodfellow.

#### About Texture
- [ ] [Image style transfer using convolutional neural networks. In: CVPR. (2016)]()
- [ ] [Ulyanov, D., Lebedev, V., Vedaldi, A., Lempitsky, V.: Texture networks: Feed-forward synthesis of textures and stylized images. In: ICML. (2016)]()
- [ ] [Johnson, J., Alahi, A., Fei-Fei, L.: Perceptual losses for real-time style transfer and superresolution. In: ECCV. (2016)]()

### WaveNet Based
- [ ] [A Wavenet for Speech Denoising.(ICASSP2018)](https://arxiv.org/abs/1706.07162)
	* An end-to-end learning method for speech denoising based on Wavenet.
- [x] [A Universal Music Translation Network(FAIR. 2018,May 21th)](https://github.com/fancoo/Papers-Reading/blob/master/Voice-Transfer/A%20Universal%20Music%20Translation%20Network.pdf):star::star::star::star:
	* Use WaveNet autoencoder to translate music across musical instruments, genres, and styles. All instruments share the same encoder, but with different decoder.
	* Two major loss, one is for the loss between decoder resconstruction with the ground-truth. the other is an instrument classification loss.
	* The results can be listened on [youtube](https://www.youtube.com/watch?v=vdxCqNWTpUs). Though the transfer result is not as good as human musician for known voice, for the unknown voice(like whistling) the transfer results is even better than human. (Maybe because human are not so familiar with the melody ?)
	* They distance their work with Style Transfer, because they believe that **a melody played by a piano is not similar except for audio texture differences to the same melody sung by a chorus**
- [ ] [Neural Audio Synthesis of Musical Notes with WaveNet Autoencoders(Submitted on 5 Apr 2017)](https://arxiv.org/abs/1704.01279)

### VAE & GAN for Speech
- [ ] [van den Oord, A., Vinyals, O., kavukcuoglu, k.: Neural Discrete Representation Learning. In: NIPS. (2017)](https://arxiv.org/abs/1711.00937)
	*  Vector Quantised-Variational AutoEncoder (VQ-VAE)
- [ ] [Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks]()
	* Use CycleGAN for `Voice Conversion`.

## License
This project is licensed under the terms of the MIT license.


