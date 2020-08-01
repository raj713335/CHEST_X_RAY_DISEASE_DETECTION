import keras
from keras import backend as K
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import sklearn
import lifelines
import shap


from util import *

# This sets a common size for all the figures we will draw.
plt.rcParams['figure.figsize'] = [10, 7]




model = load_C3M3_model()



IMAGE_DIR = 'nih_new/images-small/'
df = pd.read_csv("nih_new/train-small.csv")
im_path = IMAGE_DIR + '00016650_000.png'
x = load_image(im_path, df, preprocess=False)
plt.imshow(x, cmap = 'gray')
plt.show()



mean, std = get_mean_std_per_batch(df)



labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Atelectasis',
              'Pneumothorax', 'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']

processed_image = load_image_normalize(im_path, mean, std)
preds = model.predict(processed_image)
pred_df = pd.DataFrame(preds, columns = labels)
pred_df.loc[0, :].plot.bar()
plt.title("Predictions")
plt.show()




spatial_maps =  model.get_layer('conv5_block16_concat').output
print(spatial_maps)



get_spatial_maps = K.function([model.input], [spatial_maps])
print(get_spatial_maps)



# get an image
x = load_image_normalize(im_path, mean, std)
print(f"x is of type {type(x)}")
print(f"x is of shape {x.shape}")



# get the spatial maps layer activations (a list of numpy arrays)
spatial_maps_x_l = get_spatial_maps([x])

print(f"spatial_maps_x_l is of type {type(spatial_maps_x_l)}")
print(f"spatial_maps_x_l is has length {len(spatial_maps_x_l)}")




# get the 0th item in the list
spatial_maps_x = spatial_maps_x_l[0]
print(f"spatial_maps_x is of type {type(spatial_maps_x)}")
print(f"spatial_maps_x is of shape {spatial_maps_x.shape}")



# Get rid of the batch dimension
spatial_maps_x = spatial_maps_x[0] # equivalent to spatial_maps_x[0,:]
print(f"spatial_maps_x without the batch dimension has shape {spatial_maps_x.shape}")
print("Output some of the content:")
print(spatial_maps_x[0])





# get the output of the model
output_with_batch_dim = model.output
print(f"Model output includes batch dimension, has shape {output_with_batch_dim.shape}")



# Get the output without the batch dimension
output_all_categories = output_with_batch_dim[0]
print(f"The output for all 14 categories of disease has shape {output_all_categories.shape}")



# Get the first category's output (Cardiomegaly) at index 0
y_category_0 = output_all_categories[0]
print(f"The Cardiomegaly output is at index 0, and has shape {y_category_0.shape}")



# Get gradient of y_category_0 with respect to spatial_maps

gradient_l = K.gradients(y_category_0, spatial_maps)
print(f"gradient_l is of type {type(gradient_l)} and has length {len(gradient_l)}")

# gradient_l is a list of size 1.  Get the gradient at index 0
gradient = gradient_l[0]
print(gradient)




# Create the function that gets the gradient
get_gradient = K.function([model.input], [gradient])
type(get_gradient)



# get an input x-ray image
x = load_image_normalize(im_path, mean, std)
print(f"X-ray image has shape {x.shape}")






# use the get_gradient function to get the gradient (pass in the input image inside a list)
grad_x_l = get_gradient([x])
print(f"grad_x_l is of type {type(grad_x_l)} and length {len(grad_x_l)}")

# get the gradient at index 0 of the list.
grad_x_with_batch_dim = grad_x_l[0]
print(f"grad_x_with_batch_dim is type {type(grad_x_with_batch_dim)} and shape {grad_x_with_batch_dim.shape}")

# To remove the batch dimension, take the value at index 0 of the batch dimension
grad_x = grad_x_with_batch_dim[0]
print(f"grad_x is type {type(grad_x)} and shape {grad_x.shape}")

print("Gradient grad_x (show some of its content:")
print(grad_x[0])



# Use K.function to generate a single function
# Notice that a list of two tensors, is passed in as the second argument of K.function()
get_spatial_maps_and_gradient = K.function([model.input], [spatial_maps, gradient])
print(type(get_spatial_maps_and_gradient))





# The returned function returns a list of the evaluated tensors
tensor_eval_l = get_spatial_maps_and_gradient([x])
print(f"tensor_eval_l is type {type(tensor_eval_l)} and length {len(tensor_eval_l)}")





# store the two numpy arrays from index 0 and 1 into their own variables
spatial_maps_x_with_batch_dim, grad_x_with_batch_dim = tensor_eval_l
print(f"spatial_maps_x_with_batch_dim has shape {spatial_maps_x_with_batch_dim.shape}")
print(f"grad_x_with_batch_dim has shape {grad_x_with_batch_dim.shape}")



# Note: you could also do this directly from the function call:
spatial_maps_x_with_batch_dim, grad_x_with_batch_dim = get_spatial_maps_and_gradient([x])
print(f"spatial_maps_x_with_batch_dim has shape {spatial_maps_x_with_batch_dim.shape}")
print(f"grad_x_with_batch_dim has shape {grad_x_with_batch_dim.shape}")





# Remove the batch dimension by taking the 0th index at the batch dimension
spatial_maps_x = spatial_maps_x_with_batch_dim[0]
grad_x = grad_x_with_batch_dim[0]
print(f"spatial_maps_x shape {spatial_maps_x.shape}")
print(f"grad_x shape {grad_x.shape}")

print("\nSpatial maps (print some content):")
print(spatial_maps_x[0])
print("\nGradient (print some content:")
print(grad_x[0])


# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def grad_cam(input_model, image, category_index, layer_name):
    """
    GradCAM method for visualizing input saliency.

    Args:
        input_model (Keras.model): model to compute cam for
        image (tensor): input to model, shape (1, H, W, 3)
        cls (int): class to compute cam with respect to
        layer_name (str): relevant layer in model
        H (int): input height
        W (int): input width
    Return:
        cam ()
    """
    cam = 0

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # 1. Get placeholders for class output and last layer
    # Get the model's output
    output_with_batch_dim = input_model.output

    # Remove the batch dimension
    output_all_categories = output_with_batch_dim[0]

    # Retrieve only the disease category at the given category index
    y_c = output_all_categories[category_index]

    # Get the input model's layer specified by layer_name, and retrive the layer's output tensor
    spatial_map_layer = input_model.get_layer(layer_name).output

    # 2. Get gradients of last layer with respect to output

    # get the gradients of y_c with respect to the spatial map layer (it's a list of length 1)
    grads_l = K.gradients(y_c, spatial_map_layer)

    # Get the gradient at index 0 of the list
    grads = grads_l[0]

    # 3. Get hook for the selected layer and its gradient, based on given model's input
    # Hint: Use the variables produced by the previous two lines of code
    spatial_map_and_gradient_function = K.function([input_model.input], [spatial_map_layer, grads])

    # Put in the image to calculate the values of the spatial_maps (selected layer) and values of the gradients
    spatial_map_all_dims, grads_val_all_dims = spatial_map_and_gradient_function([image])

    # Reshape activations and gradient to remove the batch dimension
    # Shape goes from (B, H, W, C) to (H, W, C)
    # B: Batch. H: Height. W: Width. C: Channel
    # Reshape spatial map output to remove the batch dimension
    spatial_map_val = spatial_map_all_dims[0]

    # Reshape gradients to remove the batch dimension
    grads_val = grads_val_all_dims[0]

    # 4. Compute weights using global average pooling on gradient
    # grads_val has shape (Height, Width, Channels) (H,W,C)
    # Take the mean across the height and also width, for each channel
    # Make sure weights have shape (C)
    weights = np.mean(grads_val, axis=(0, 1))

    # 5. Compute dot product of spatial map values with the weights
    cam = np.dot(spatial_map_val, weights)

    ### END CODE HERE ###

    # We'll take care of the postprocessing.
    H, W = image.shape[1], image.shape[2]
    cam = np.maximum(cam, 0)  # ReLU so we only get positive importance
    cam = cv2.resize(cam, (W, H), cv2.INTER_NEAREST)
    cam = cam / cam.max()

    return cam


im = load_image_normalize(im_path, mean, std)
cam = grad_cam(model, im, 5, 'conv5_block16_concat') # Mass is class 5

# Loads reference CAM to compare our implementation with.
reference = np.load("reference_cam.npy")
error = np.mean((cam-reference)**2)

print(f"Error from reference: {error:.4f}, should be less than 0.05")



plt.imshow(load_image(im_path, df, preprocess=False), cmap='gray')
plt.title("Original")
plt.axis('off')

plt.show()

plt.imshow(load_image(im_path, df, preprocess=False), cmap='gray')
plt.imshow(cam, cmap='magma', alpha=0.5)
plt.title("GradCAM")
plt.axis('off')
plt.show()


# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def compute_gradcam(model, img, mean, std, data_dir, df,
                    labels, selected_labels, layer_name='conv5_block16_concat'):
    """
    Compute GradCAM for many specified labels for an image.
    This method will use the `grad_cam` function.

    Args:
        model (Keras.model): Model to compute GradCAM for
        img (string): Image name we want to compute GradCAM for.
        mean (float): Mean to normalize to image.
        std (float): Standard deviation to normalize the image.
        data_dir (str): Path of the directory to load the images from.
        df(pd.Dataframe): Dataframe with the image features.
        labels ([str]): All output labels for the model.
        selected_labels ([str]): All output labels we want to compute the GradCAM for.
        layer_name: Intermediate layer from the model we want to compute the GradCAM for.
    """
    img_path = data_dir + img
    preprocessed_input = load_image_normalize(img_path, mean, std)
    predictions = model.predict(preprocessed_input)
    print("Ground Truth: ", ", ".join(np.take(labels, np.nonzero(df[df["Image"] == img][labels].values[0]))[0]))

    plt.figure(figsize=(15, 10))
    plt.subplot(151)
    plt.title("Original")
    plt.axis('off')
    plt.imshow(load_image(img_path, df, preprocess=False), cmap='gray')

    j = 1

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    # Loop through all labels
    for i in range(len(labels)):  # complete this line
        # Compute CAM and show plots for each selected label.

        # Check if the label is one of the selected labels
        if labels[i] in selected_labels:  # complete this line

            # Use the grad_cam function to calculate gradcam
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)

            ### END CODE HERE ###

            print("Generating gradcam for class %s (p=%2.2f)" % (labels[i], round(predictions[0][i], 3)))
            plt.subplot(151 + j)
            plt.title(labels[i] + ": " + str(round(predictions[0][i], 3)))
            plt.axis('off')
            plt.imshow(load_image(img_path, df, preprocess=False), cmap='gray')
            plt.imshow(gradcam, cmap='magma', alpha=min(0.5, predictions[0][i]))
            j += 1





df = pd.read_csv("nih_new/train-small.csv")

image_filename = '00016650_000.png'
labels_to_show = ['Cardiomegaly', 'Mass', 'Edema']
compute_gradcam(model, image_filename, mean, std, IMAGE_DIR, df, labels, labels_to_show)




image_filename = '00005410_000.png'
compute_gradcam(model, image_filename, mean, std, IMAGE_DIR, df, labels, labels_to_show)


image_name = '00004090_002.png'
compute_gradcam(model, image_name, mean, std, IMAGE_DIR, df, labels, labels_to_show)


rf = pickle.load(open('nhanes_rf.sav', 'rb')) # Loading the model
test_df = pd.read_csv('nhanest_test.csv')
test_df = test_df.drop(test_df.columns[0], axis=1)
X_test = test_df.drop('y', axis=1)
y_test = test_df.loc[:, 'y']
cindex_test = cindex(y_test, rf.predict_proba(X_test)[:, 1])

print("Model C-index on test: {}".format(cindex_test))



X_test_risky = X_test.copy(deep=True)
X_test_risky.loc[:, 'risk'] = rf.predict_proba(X_test)[:, 1] # Predicting our risk.
X_test_risky = X_test_risky.sort_values(by='risk', ascending=False) # Sorting by risk value.
X_test_risky.head()


# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def permute_feature(df, feature):
    """
    Given dataset, returns version with the values of
    the given feature randomly permuted.

    Args:
        df (dataframe): The dataset, shape (num subjects, num features)
        feature (string): Name of feature to permute
    Returns:
        permuted_df (dataframe): Exactly the same as df except the values
                                of the given feature are randomly permuted.
    """
    permuted_df = df.copy(deep=True)  # Make copy so we don't change original df

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # Permute the values of the column 'feature'
    permuted_features = np.random.permutation(permuted_df[feature])

    # Set the column 'feature' to its permuted values.
    permuted_df[feature] = permuted_features

    ### END CODE HERE ###

    return permuted_df






print("Test Case")

example_df = pd.DataFrame({'col1': [0, 1, 2], 'col2':['A', 'B', 'C']})
print("Original dataframe:")
print(example_df)
print("\n")

print("col1 permuted:")
print(permute_feature(example_df, 'col1'))

print("\n")
print("Compute average values over 1000 runs to get expected values:")
col1_values = np.zeros((3, 1000))
np.random.seed(0) # Adding a constant seed so we can always expect the same values and evaluate correctly.
for i in range(1000):
    col1_values[:, i] = permute_feature(example_df, 'col1')['col1'].values

print("Average of col1: {}, expected value: [0.976, 1.03, 0.994]".format(np.mean(col1_values, axis=1)))


# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
def permutation_importance(X, y, model, metric, num_samples=100):
    """
    Compute permutation importance for each feature.

    Args:
        X (dataframe): Dataframe for test data, shape (num subject, num features)
        y (np.array): Labels for each row of X, shape (num subjects,)
        model (object): Model to compute importances for, guaranteed to have
                        a 'predict_proba' method to compute probabilistic
                        predictions given input
        metric (function): Metric to be used for feature importance. Takes in ground
                           truth and predictions as the only two arguments
        num_samples (int): Number of samples to average over when computing change in
                           performance for each feature
    Returns:
        importances (dataframe): Dataframe containing feature importance for each
                                 column of df with shape (1, num_features)
    """

    importances = pd.DataFrame(index=['importance'], columns=X.columns)

    # Get baseline performance (note, you'll use this metric function again later)
    baseline_performance = metric(y, model.predict_proba(X)[:, 1])

    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###

    # Iterate over features (the columns in the importances dataframe)
    for feature in importances.columns:  # complete this line

        # Compute 'num_sample' performances by permutating that feature

        # You'll see how the model performs when the feature is permuted
        # You'll do this num_samples number of times, and save the performance each time
        # To store the feature performance,
        # create a numpy array of size num_samples, initialized to all zeros
        feature_performance_arr = np.zeros(num_samples)

        # Loop through each sample
        for i in range(num_samples):  # complete this line

            # permute the column of dataframe X
            perm_X = permute_feature(X, feature)

            # calculate the performance with the permuted data
            # Use the same metric function that was used earlier
            feature_performance_arr[i] = metric(y, model.predict_proba(perm_X)[:, 1])

        # Compute importance: absolute difference between
        # the baseline performance and the average across the
        importances[feature]['importance'] = np.abs(baseline_performance - np.mean(feature_performance_arr))

    ### END CODE HERE ###

    return importances


print("Test Case")
print("\n")
print("We check our answers on a Logistic Regression on a dataset")
print("where y is given by a sigmoid applied to the important feature.")
print("The unimportant feature is random noise.")
print("\n")
example_df = pd.DataFrame({'important': np.random.normal(size=(1000)), 'unimportant':np.random.normal(size=(1000))})
example_y = np.round(1 / (1 + np.exp(-example_df.important)))
example_model = sklearn.linear_model.LogisticRegression(fit_intercept=False).fit(example_df, example_y)

example_importances = permutation_importance(example_df, example_y, example_model, cindex, num_samples=100)
print("Computed importances:")
print(example_importances)
print("\n")
print("Expected importances (approximate values):")
print(pd.DataFrame({"important": 0.50, "unimportant": 0.00}, index=['importance']))
print("If you round the actual values, they will be similar to the expected values")



importances = permutation_importance(X_test, y_test, rf, cindex, num_samples=100)
print(importances)



importances.T.plot.bar()
plt.ylabel("Importance")
l = plt.legend()
l.remove()
plt.show()




explainer = shap.TreeExplainer(rf)
i = 0 # Picking an individual
shap_value = explainer.shap_values(X_test.loc[X_test_risky.index[i], :])[1]
shap.force_plot(explainer.expected_value[1], shap_value, feature_names=X_test.columns, matplotlib=True)



shap_values = shap.TreeExplainer(rf).shap_values(X_test)[1]



shap.summary_plot(shap_values, X_test)


shap.dependence_plot('Age', shap_values, X_test, interaction_index = 'Sex')



shap.dependence_plot('Poverty index', shap_values, X_test, interaction_index='Age')














