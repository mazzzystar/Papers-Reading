# Papers-Reading
My reading notes on DL papers, along with my personal comment of each paper, so there may exist lots of mistakes, I really appreciate you to point out.

## Contents
### Neural Style Transfer
- [x] [Neural Style Transfer: A Review](https://github.com/fancoo/Papers-Reading/blob/master/Neural-Style-Transfer/Neural%20Style%20Transfer-A%20Review.pdf) :star::star::star::star:
	* Investigate the works of Neural Style Transfer till May of 2016.
- [ ] [Demystifying Neural Style Transfer](https://arxiv.org/abs/1701.01036)
	* Prove that matching the Gram matrices is actually equivalent to minimize the Maximum Mean Discrepancy(MMD) with second order polynomial kernel.
	* Try out for different kernels and parameters.
- [ ] [Fast Patch-based Style Transfer of Arbitrary Style](https://arxiv.org/abs/1612.04337)
	* A more advanced version of "Fast" Neural Style Transfer that can run in real-time and applies to infinite kind of styles.
	* The drawback is the quality of stylized images is worse than "Fast" Neural Style which yet can only applies to finite styles.

### Generative Model
- [x] [Pixel Recurrent Neural Networks(Best Paper of ICML2016)](https://github.com/fancoo/Papers-Reading/blob/master/Generative-Model/Pixel%20Recurrent%20Neural%20Networks.pdf) :star::star::star::star:
	* I quickly skimmed this paper, it introduced a new method to generate image pixel by pixel with sequence model, which means **you can only predict current pixel by it's previous pixels(namely the pixels above and to the left of it).** To achieve this, they introduce a `mask` to make sure model can not read later pixels.
	* The loss curve is much more smooth and interpretatable compared to GAN.
- [x] [Conditional Image Generation with PixelCNN Decoders](https://github.com/fancoo/Papers-Reading/blob/master/Generative-Model/Conditional%20Image%20Generation%20with%20PixelCNN%20Decoders.pdf) :star::star::star::star::star:
	* An improvement to PixelRNN & PixelCNN by adding an additional `Gated activation unit`.
	* Use two stack(vertical and horizontal) to aviod the `blind spot` in Mask.
	* Explore the performance of image generation in this kind of `Gated PixelCNN` in conditional distribution image, actually it seems not as good as GAN but, still another method and therefore lead to the famous [WaveNet](https://arxiv.org/abs/1609.03499).
- [ ] [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) :star::star::star::star::star:
	* A summary of papers above, and use these methods in audio.
	* Keywords: fuse the technic of `Dilated Casual Convolution`, `Gated Activation Units` and `residual network` along with `skip connections`.
	* Based on `Conditional WaveNet`, they explored `Multi-Speaker Speech Generation`, `TTS(Text-To-Speech)` and `Music Generation` by feeding additional input `h`. In speech generation, it's speaker ID of one-hot vector, in TTS it's the text, in music generation it's the tag of generated musich, like the instruments or the genre.
- [ ] [Parallel WaveNet: Fast High-Fidelity Speech Synthesis](https://arxiv.org/abs/1711.10433)


## License
This project is licensed under the terms of the MIT license.


