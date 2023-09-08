println("starting")

function logsig(x)
    y = 1/(1+exp(-x));
    return y
end

function read_MNIST_images(filename)
    fid = open(filename)

    #magic number
    #The first 2 bytes are always 0
    read(fid,2)
    #The third byte codes the type of the data:
    #0x08: unsigned byte, etc...
    data_type = read(fid,1)
    #The 4-th byte codes the number of dimensions of the vector/matrix: 1 for
    #vectors, 2 for matrices....
    number_of_dimensions = read(fid,1);
    #size in dimension 0:
    N_images = ntoh(read(fid,Int32));
    #size in dimension 1:
    N_lines = ntoh(read(fid,Int32));
    #size in dimension 2:
    N_cols = ntoh(read(fid,Int32));
    
    pixel = zeros(UInt8, N_lines, N_cols, N_images);
    for image in 1:N_images
        for line in 1:N_lines
            for col in 1:N_cols
                a = read(fid,1);
                pixel[line, col, image] = a[1];
            end
        end
    end
    close(fid)
    return pixel;
end

function read_MNIST_labels(filename)
    fid = open(filename)
    #magic number
    #The first 2 bytes are always 0
    read(fid,2)
    #The third byte codes the type of the data:
    #0x08: unsigned byte, etc...
    data_type=read(fid,1);
    n_dims = read(fid,1);
    n_labels = ntoh(read(fid,Int32));
    labels = zeros(UInt8,1,n_labels);
    for i in 1:n_labels
        a = read(fid,1)
        labels[i] = a[1];
    end
    close(fid);
    return labels;
end

#I'm only using the train images and labels because the test-labels file
#is in a format that I don't have documentation for
#of course I still separate train and test images later
cd()
pixels= read_MNIST_images(pwd() * "/train-images.idx3-ubyte");
labels = read_MNIST_labels(pwd() * "/train-labels.idx1-ubyte");

#present input images as vectors and normalize
X = reshape(pixels,(784,60000));
X = X/255;

#constitute output vectors:
T = zeros(10,60000);
for i in 1:60000
    T[labels[i]+1, i] = 1;
end

#separate train and test
X_train = X[:,1:50000];
X_test = X[:,50001:60000];
T_train = T[:,1:50000];
T_test = T[:,50001:60000];

#generate MLP:
ILU = 784;
HLU = 10;
OLU = 10;
W1 = randn(HLU,ILU);
B1 = randn(HLU,1);
W2 = randn(OLU,HLU);
B2 = randn(OLU,1);

#start training:
N_epochs = 10;
errors = zeros(1,N_epochs);
for epoch in 1:N_epochs
    print("epoch ",epoch)
    for i = 1:50000
        Y = logsig.(W1*X_train[:,i] .+ B1);
        O = logsig.(W2*Y .+ B2);

        answer = findmax(O);
        answer = answer[2][1];
        correct = findmax(T_train[:,i]);
        correct = correct[2][1];
        if answer != correct
            global errors[epoch] = errors[epoch] + 1;
        end

        d = 2*(O .- T_train[:,i]).*O.*(1 .- O)/OLU;
        dEdB2 = d;
        dEdW2 = d*transpose(Y);
        d = (transpose(W2)*d) .*Y.*(1 .- Y);
        dEdB1 = d;
        dEdW1 = d*transpose(X[:,i]);

        global B2 = B2 - 0.1*dEdB2;
        global W2 = W2 - 0.1*dEdW2;
        global B1 = B1 - 0.1*dEdB1;
        global W1 = W1 - 0.1*dEdW1;
    end
    println(" error rate: ", errors[epoch]*100/50000)
end

#testing the network on unseen data:
errors = 0
for i = 1:10000
    Y = logsig.(W1*X_test[:,i] .+ B1);
    O = logsig.(W2*Y .+ B2);
    
    answer = findmax(O);
    answer = answer[2][1];
    correct = findmax(T_test[:,i]);
    correct = correct[2][1];
    if answer != correct
        global errors = errors + 1;
    end
end
println(" error rate on test set: ", errors*100/10000)

println("done")