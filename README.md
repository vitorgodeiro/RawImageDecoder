# RawImageDecoder

<p align="center">
<img src="https://github.com/vitorgodeiro/RawImageDecoder/blob/master/imgReport/wb_grayworldGama.png" width="200"><img src="https://github.com/vitorgodeiro/RawImageDecoder/blob/master/imgReport/wb_whitePatchGama.png" width="200"><img src="https://github.com/vitorgodeiro/RawImageDecoder/blob/master/imgReport/wb_manualGama.png" width="200"><img src="https://github.com/vitorgodeiro/RawImageDecoder/blob/master/imgReport/wb_percentil_gama.png" width="200">  
 </p>
 This is the image processing module of a digital camera has to convert the capture raw image data into a full color image. The illustrations of using white balance algorithm Gray World, White Patch, Iterative and Percentile respectively.

## About

This project is a raw image decoder implemented in Python. This code convert the raw image data into full color image. In this pipeline we read file in [R G G B] pattern and perform the bilinear demosaic. Next, we can choose white balance algorithm between Gray World, White Patch, Iterative and Percentile. Finally code perform the gamma correction and give the result image.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

What things you need to run the software 


* Argparse
* Numpy
* OpenCV
* Python
* Rawpy
* Scipy

## Running 

```
python main.py
```

Use the flag '- path pathIMG' for run in another image.
## Author

* **Vítor Godeiro**

## License
This project is licensed under MIT license - see the [LICENSE](LICENSE) file for details.
