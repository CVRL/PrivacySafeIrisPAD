# Privacy-Safe Iris Presentation Attack Detection

Official repository for the paper: Mahsa Mitcheff, Patrick Tinsley, Adam Czajka, "Privacy-Safe Iris Presentation Attack Detection," IEEE/IAPR International Joint Conference on Biometrics (IJCB), September 15-18, 2024, Buffalo, NY, USA 

Paper: IEEEXplore | ArXiv pre-print

## Abstract

This paper and repository propose a framework for a privacy-safe iris presentation attack detection (PAD) method, designed solely with synthetically-generated, identity-leakage-free iris images. Once trained, the method is evaluated in a classical way using state-of-the-art iris PAD benchmarks. We designed two generative models for the synthesis of ISO/IEC 19794-6-compliant iris images. The first model synthesizes bona fide-looking samples. To avoid “identity leakage,” the generated samples that accidentally matched those used in the model’s training were excluded. The second model synthesizes images of irises with textured contact lenses and is conditioned by a given contact lens brand to have better control over textured contact lens appearance when forming the training set. Our experiments demonstrate that models trained solely on synthetic data achieve a slightly lower but still reasonable performance when compared to solutions trained with iris images collected from human subjects. This is the first-of-its-kind attempt to use solely synthetic data to train a fully-functional iris PAD solution, and despite the performance gap between regular and the proposed methods, this study demonstrates that with the increasing fidelity of generative models, creating such privacy-safe iris PAD methods may be possible.

## Installation and Usage

...

## Citations

IJCB 2024 paper:

```
@InProceedings{Mitcheff_IJCB_2024,
  author    = {Mahsa Mitcheff and Patrick Tinsley and Adam Czajka},
  booktitle = {The IEEE/IAPR International Joint Conference on Biometrics (IJCB)},
  title     = {{Privacy-Safe Iris Presentation Attack Detection}},
  year      = {2024},
  address   = {Buffalo, NY, USA},
  month     = {September 15-18},
  pages     = {1-8},
  publisher = {IEEE}
}
```

This GitHub repository:

```
@Misc{ND_OpenSourceIrisRecognition_GitHub,
  howpublished = {\url{https://github.com/CVRL/PrivacySafeIrisPAD/}},
  note         = {Accessed: X},
  title        = {{Privacy-Safe Iris Presentation Attack Detection (IJCB 2024 paper repository)}},
  author       = {Mahsa Mitcheff and Patrick Tinsley and Adam Czajka},
}
```

## License

...
