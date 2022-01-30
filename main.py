import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import svd1

FOLDER = "./Dataset/"
FILES = os.listdir(FOLDER)
TEST_DIR = "./Testset/"

def calculate(A):
    m = A.shape[0]
    n = A.shape[1]
    S = np.zeros(n)

    # finding eigenvectors with biggest eigenvalues of A*transpose(A)
    helper = np.dot(A.T, A)
    eigenvalues, eigenvectors = np.linalg.eigh(helper)
    # descending sort of all the eigenvectors according to their eigenvalues
    index = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[index]
    eigenvectors = eigenvectors[:, index]
    V = eigenvectors
    # S is a diagonal matrix that keeps square root of eigenvalues
    j = 0
    for i in eigenvalues:
        if j == S.size:
            break
        elif i >= 0:
            S[j] = np.sqrt(i)
            j += 1
    sigma = np.zeros((m, n))
    for i in range(m):
        sigma[i][i] = eigenvalues[i]
    U = np.empty((m, m), float)
    for i in range (m):
        U[i] = np.dot(A, V[i])/S[i]
    return U, sigma, V.T

def load_images_train_and_test(TEST):
    test=np.asarray(Image.open(TEST)).flatten()
    train=[]
    for name in FILES:
        train.append(np.asarray(Image.open(FOLDER + name)).flatten())
    train= np.array(train)
    return test,train
   
def normalize(test,train):
    """
    TODO : Normalize test and train and return them properly
    Hint : To calculate mean properly use two arguments version of mean numpy method (https://www.javatpoint.com/numpy-mean)
    Hint : normalize test with train mean
    """
    normalized_test = test - np.mean(test)
    normalized_train = train - np.mean(train)
    return normalized_test, normalized_train
    pass

def svd_function(images):
    """
    TODO : implement SVD (use np.linalg.svd) and return u,s,v 
    Additional(Emtiazi) todo : implement svd without using np.linalg.svd
    """
    # u, s, v = np.linalg.svd(images, full_matrices=False)
    u, s, v = calculate(images)
    return u, s, v
    pass

def project_and_calculate_weights(img,u):
    """
    TODO : calculate element wise multiplication of img and u . (you can use numpy methods)
    """
    multiply = np.multiply(img, u)
    return multiply
    pass

def predict(test,train):
    """
    TODO : Find the most similar face to test among train set by calculating errors and finding the face that has minimum error
    return : index of the data that has minimum error in train dataset
    Hint : error(i) = norm(train[:,i] - test)       (you can use np.linalg.norm)
    """
    x, y =train.shape
    error = np.empty(y, float)
    min_error = 999999
    min_index = -1
    for i in range(y):
        error[i] = np.linalg.norm(train[:, i] - test)
        if error[i] < min_error:
            min_error = error[i]
            min_index = i

    return min_index
    pass

def plot_face(tested,predicted):
    """
    TODO : Plot tested image and predicted image . It would be great if you show them next to each other 
    with subplot and figures that you learned in matplotlib video in the channel.
    But you are allowed to show them one by one
    """
    plt.subplot(121)
    plt.imshow(tested, cmap="gray")
    plt.subplot(122)
    plt.imshow(predicted, cmap="gray")
    plt.show()
    pass


if __name__ == "__main__":
    true_predicts=0
    all_predicts=0
    for TEST_FILE in os.listdir(TEST_DIR):
        # Loading train and test
        test, train = load_images_train_and_test(TEST_DIR+TEST_FILE)

        # Normalizing train and test
        test, train = normalize(test, train)
        test = test.T
        train = train.T
        test = np.reshape(test, (test.size, 1))
        # Singular value decomposition
        u, s, v = svd_function(train)

        # Weight for test
        w_test = project_and_calculate_weights(test, u)
        w_test = np.array(w_test, dtype='int8').flatten()

        # Weights for train set
        w_train = []
        for i in range(train.shape[1]):
            w_i = project_and_calculate_weights(np.reshape(train[:, i], (train[:, i].size, 1)),  u)
            w_i = np.array(w_i, dtype='int8').flatten()
            w_train.append(w_i)
        w_train = np.array(w_train).T

        # Predict
        index_of_most_similar_face = predict(w_test,  w_train)

        # Showing results
        print("Test : "+TEST_FILE)
        print(f"The predicted face is: {FILES[index_of_most_similar_face]}")
        print("\n***************************\n")

        # Calculating Accuracy
        all_predicts+=1
        if FILES[index_of_most_similar_face].split("-")[0]==TEST_FILE.split("-")[0]:
            true_predicts+=1
            # Plotting correct predictions
            plot_face(Image.open(TEST_DIR+TEST_FILE), Image.open(FOLDER+FILES[index_of_most_similar_face]))
        else:
            # Plotting wrong predictions
            plot_face(Image.open(TEST_DIR+TEST_FILE), Image.open(FOLDER+FILES[index_of_most_similar_face]))

    # Showing Accuracy
    accuracy=true_predicts/all_predicts
    print(f'Accuracy : {"{:.2f}".format(accuracy*100)} %')

