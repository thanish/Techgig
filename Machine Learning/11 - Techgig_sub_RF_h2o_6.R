#Office
setwd('D:\\Thanish\\D\\Thanish Folder\\Compeditions\\Techgig\\ML')

#Laptop
setwd('E:\\Thanish\\Data science\\Techgig\\ML')

transaction = read.csv('Code-Gladiators-Transaction.csv')
Inv_exp = read.csv('Code-Gladiators-InvestmentExperience.csv')
activity = read.csv('Code-Gladiators-Activity.csv')
test = read.csv('test_data.csv')

str(transaction)
str(Inv_exp)
str(activity)
str(test)

#Extracing only the unique values in activity
activity = activity[!duplicated(activity[,c('Unique_Advisor_Id','Month')]),]

#Seperating year and month
transaction$Year = as.numeric(substr(transaction$Month, start= 1, stop= 4))
transaction$Month = as.numeric(substr(transaction$Month, start= 8, stop= 9))
Inv_exp$Year = as.numeric(substr(Inv_exp$Month, start= 1, stop= 4))
Inv_exp$Month = as.numeric(substr(Inv_exp$Month, start= 8, stop= 9))
activity$Year = as.numeric(substr(activity$Month, start= 1, stop= 4))
activity$Month = as.numeric(substr(activity$Month, start= 8, stop= 9))
test$Year = 2017
test$Month = 01

#Correcting the month and year to include 2017
Inv_exp$Month = Inv_exp$Month + 1
Inv_exp$Year[Inv_exp$Month==13] = Inv_exp$Year[Inv_exp$Month==13] + 1
Inv_exp$Month[Inv_exp$Month==13] = 01
activity$Month = activity$Month + 1
activity$Year[activity$Month==13] = activity$Year[activity$Month==13] + 1
activity$Month[activity$Month==13] = 01

#Removing the columns as of now 
transaction = transaction[,c('Unique_Advisor_Id', 'Unique_Investment_Id',
                             'Year', 'Month', 'Transaction_Type')]
test$Transaction_Type = NA 
transaction$type = 'train'
test$type = 'test'
train_test_prod = rbind(transaction, test)
str(train_test_prod)

#Merging the tables
#train_test with Inv_exp
train_test_prod = merge(train_test_prod, Inv_exp,
                        by.x = c('Unique_Investment_Id', 'Year', 'Month'),
                        by.y = c('Unique_Investment_Id', 'Year', 'Month'),
                        all.x = T)

#train_test with activity
train_test_prod = merge(train_test_prod, activity,
                        by.x = c('Unique_Advisor_Id', 'Year', 'Month'),
                        by.y = c('Unique_Advisor_Id', 'Year', 'Month'),
                        all.x = T)

str(train_test_prod)

nrow(train_test_prod)

#Dropping few columns factor columns
train_test_prod$Morningstar.Category = NULL
train_test_prod$Investment = NULL
str(train_test_prod)

#Filling up the NA's 
train_test_prod[is.na(train_test_prod)] = -999

#Splitting back to train and test
train_prod = train_test_prod[train_test_prod$type=='train',]
test_prod = train_test_prod[train_test_prod$type=='test',]
train_prod$type = NULL
test_prod$type = NULL

#Splitting into local train an test
train_local = train_prod[train_prod$Month <=9,]
test_local = train_prod[train_prod$Month >9,]

#H2o
library(h2o)
h2o.init(nthreads = -1)
train_prod_h2o = as.h2o(train_prod)
train_local_h2o = as.h2o(train_local)
test_local_h2o = as.h2o(test_local)
test_prod_h2o = as.h2o(test_prod)

x_indep = setdiff(colnames(train_local_h2o), 'Transaction_Type')
y_dep = 'Transaction_Type'


#RF
RF.model.local = h2o.randomForest(x = x_indep, y = y_dep, 
                                  ntrees = 200, max_depth = 25,
                                  training_frame = train_prod_h2o,
                                  seed=100)
RF.local.pred = h2o.predict(RF.model.local, newdata = test_local_h2o)
RF.local.pred = as.data.frame(RF.local.pred)
RF_table = table(RF.local.pred$R>0.4, test_local$Transaction_Type)
sum(diag(RF_table))/nrow(test_local)

h2o.varimp_plot(RF.model.local)

RF.prod.pred = h2o.predict(RF.model.local, newdata = test_prod_h2o)
RF.prod.pred = as.data.frame(RF.prod.pred)
RF.prod.pred_sum = ifelse(RF.prod.pred$R>0.4, "YES", "NO")
sub_RF_H2o = data.frame(Unique_Advisor_Id= test_prod$Unique_Advisor_Id,
                        Unique_Investment_Id= test_prod$Unique_Investment_Id,
                        Propensity_Score= RF.prod.pred$R,
                        Redeem_Status= RF.prod.pred_sum)

write.csv(sub_RF_H2o, row.names = F, '11-sub_RF_h2o_6.csv')


