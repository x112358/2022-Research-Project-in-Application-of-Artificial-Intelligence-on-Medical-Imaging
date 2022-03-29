# 2022-Research-Project-in-Application-of-Artificial-Intelligence-on-Medical-Imaging
# 1.	Introduction 
Detect Pneumonia from chest X-ray images by using deep learning. There are two kinds of labels, normal and pneumonia. Implement dl model architecture and show the accuracy.
# 2.	Experiment setups 
a.	The detail of your model 
Architecture: Desnet161
Loss: Cross Entropy
Learning rate: 1e-4
Optimizer: Adam
Weight decay: 1e-4
b.	The detail of you Dataloder 
Batch size = 10
Preprocess:
transform.Resize(255),
transform.CenterCrop(224),
transform.RandomHorizontalFlip(),
transform.RandomRotation(10),
transform.RandomGrayscale(),
transform.RandomAffine(translate=(0.05,0.05), degrees=0),
transform.ToTensor()

# 3.	Experiment result
a.	Highest testing accuracy and F1-score (Screenshot) 
 ![score](https://user-images.githubusercontent.com/23052407/160682748-55f0cd6b-37d8-41da-a83a-78679e2767d2.PNG)

b.	Ploting the comparsion figure 
 ![performance](https://user-images.githubusercontent.com/23052407/160682756-b61f809a-0d6d-4f59-98da-fb95f3722666.png)

 ![confusion_matrix](https://user-images.githubusercontent.com/23052407/160682727-236ba0f1-0e98-4326-8ace-97b5b5e97eec.png)

c. Anything you want to present 
the distribution of normal and pneumonia isn’t balanced.
# 4.	Discussion 
a.	Anything you want to share 
The best accuracy is beyond 98%, which means that desnet161 is a good architecture for detecting pneumonia from chest X-ray images

