#Reading the files
train_prod = read.csv('train_data.csv')
test_prod = read.csv('test_data.csv')
str(train_prod)

#Dropping the blanks in the train prod
train_prod = train_prod[!is.na(train_prod$is_fraud),]
str(train_prod)

#Adding is_fraud to test
test_prod$is_fraud = NA
train_prod$type = 'train'
test_prod$type = 'test'

#Changing the colnames to match both test and train
colnames(train_prod) = colnames(test_prod)  

#Combining the train and test
train_test_prod = rbind(train_prod, test_prod)

#Converting Date feature
train_test_prod$Date = substr(train_test_prod$click_time, start = 1, stop = 10)
train_test_prod$Day = as.numeric(substr(train_test_prod$Date, start = 9, stop = 10))
train_test_prod$Time = substr(train_test_prod$click_time, start = 12, stop = 23)
train_test_prod$Hour = as.numeric(substr(train_test_prod$Time, start = 1, stop = 2))
train_test_prod$Minute = as.numeric(substr(train_test_prod$Time, start = 4, stop = 5))

#Dropping of the column 
train_test_prod$click_time = NULL
train_test_prod$Date = NULL
train_test_prod$Time = NULL

#Extracting only the factor features
fac_columns = colnames(train_test_prod)[sapply(train_test_prod, class)=='factor']
fac_columns = setdiff(fac_columns,'is_fraud')

#Dropping the factor as of now
train_test_prod = train_test_prod[,setdiff(colnames(train_test_prod),fac_columns)]

#Splitting it back to train an test
train_prod = train_test_prod[train_test_prod$type=='train',]
test_prod = train_test_prod[train_test_prod$type=='test',]
train_prod$type = NULL
test_prod$type = NULL
test_prod$is_fraud = NULL
str(test_prod)

#Splitting into local train and test
set.seed(100)
split = sample(nrow(train_prod), 0.6*nrow(train_prod), replace = F)
train_local = train_prod[split,]
test_local = train_prod[-split,]

#Converting the target column to be factors
train_local$is_fraud = as.factor(train_local$is_fraud)
test_local$is_fraud = as.factor(test_local$is_fraud)

#H2o
library(h2o)
h2o.init(nthreads = -1)
train_local_h2o = as.h2o(train_local)
test_local_h2o = as.h2o(test_local)
test_prod_h2o = as.h2o(test_prod)

x_indep = setdiff(colnames(train_local_h2o), 'is_fraud')
y_dep = 'is_fraud'

#GBM
gbm.model.local = h2o.gbm(x = x_indep, y = y_dep, 
                          ntrees = 200, max_depth = 15,
                          training_frame = train_local_h2o,
                          seed=100)
gbm.local.pred = h2o.predict(gbm.model.local, newdata = test_local_h2o)
gbm.local.pred = as.data.frame(gbm.local.pred)
gbm_table = table(gbm.local.pred$p1>0.5, test_local$is_fraud)
sum(diag(gbm_table))/nrow(test_local)

h2o.varimp_plot(gbm.model.local)

#On test prod
gbm.prod.pred = h2o.predict(gbm.model.local, newdata = test_prod_h2o)
gbm.prod.pred = as.data.frame(gbm.prod.pred)

gbm.prod.pred_sum = ifelse(gbm.prod.pred$p1>0.5, 1, 0)
test_prod = readRDS('test_prod.rds')
sub_gbm_H2o = cbind(test_prod, 
                    fraud_click_probability = gbm.prod.pred$p1,	
                    is_fraud = gbm.prod.pred_sum)

write.csv(sub_gbm_H2o, row.names = F, 'sub_gbm_h2o_1.csv')
