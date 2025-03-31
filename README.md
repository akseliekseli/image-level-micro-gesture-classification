# image-level-micro-gesture-classification
LUT University Machine Vision and Digital Image Analysis Spring Term 2025 - Practical assignment

Create venv:
```python3 -m venv venv```
Activate:
```source venv/bin/activate```

Install required packages:
```pip install -r requirements.txt```

Running the code for training:

```python3 train-model.py --model swin --gpu 2 --epochs 30```
where model: swin, resnet, pose
    gpu: number of which gpu to use (cuda:0, cuda:1 etc.)
    epoch: number of training epochs

And for testing:

``` python3 test-model.py --model swin --test_dir test/```
for same model parameters as above and directory of the test samples.

For the 30% of the best accuracies, the Swin model should be used in our case. 
