dataset = read.csv('Salary_Data.csv')




#install.packages('caTools')
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3)
training_set = subset(dataset,split == TRUE)
test_set = subset(dataset,split == FALSE)


#feature scaling
# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])