'''
此次项目代码分为三大部分、
(1) 数据预处理(数据清洗、多源数据融合等);
(2) 实体嵌入(Entity Eembedding)处理数据；
(3) 构建深度学习/机器学习模型来实证预测。
'''

#########################################################(1)数据清洗#####################################################################

# 导入需要使用各数据包
library(data.table)
library(dplyr)
library(VIM)
library(lubridate)
library(car)
options(scipen = 100)


# 导入并初步清洗数据，选点青岛市作为研究目标
tobacoo <- fread('E:\\project\\consumers.csv', encoding='UTF-8')
tobacoo <- tobacoo[-which(tobacoo$open_id=='null'), ]
tobacoo <- tobacoo[-which(is.na(tobacoo$longitude)), ]  
sum(is.na(tobacoo$longitude))
sum(is.na(tobacoo$latitude))
tobacoo <- tobacoo[, -c(2,9)]
tobacoo <- filter(tobacoo, city=='青岛市')



# 将消费者消费次数合并至初始数据集
open_id_Freq<- data.table(table(tobacoo$open_id))
tobacoo_rep <- merge(tobacoo, open_id_Freq, by.x = 'open_id', by.y = 'V1', all.x = T)
names(tobacoo_rep)[12] <- c('Frequency')
tobacoo_rep$Frequency <- as.numeric(tobacoo_rep$Frequency)



# 构建RFM模型指标(F已有，即Frequency这一列)
# Recency
tobacoo_rep$Recency <- NA 
tobacoo_rep$crt_date <- tobacoo_rep$crt_date %>% ymd()
summary(tobacoo_rep$crt_date)
tobacoo_rep$Recency <- difftime(max(tobacoo_rep$crt_date),tobacoo_rep$crt_date, units="days") %>% round() %>% as.numeric()
tobacoo_rep[,c(12, 13)] <- tobacoo_rep[, c(13, 12)] # 改变列的顺序
names(tobacoo_rep)[12:13] <- c('Recency', 'Frequency')



# 截取最后一个月有1次以上消费记录的消费者
range(tobacoo_rep$crt_date)
date_1 <-as.POSIXct('2018-06-20') 
date_2 <- as.POSIXct('2018-07-21')
last_month <- interval(start = date_1, end = date_2)
last_customer <- tobacoo_rep[ymd(tobacoo_rep$crt_date) %within% last_month, ]
unique(last_customer$open_id)  # 22255个消费者
last_customer_df<- data.table(table(last_customer$open_id))



# 划分数据集
# 训练集
date_3 <- as.POSIXct('2017-06-29')
date_4 <- as.POSIXct('2018-06-20')
before_month <- interval(start = date_3, end = date_4)
before_customer <- tobacoo_rep[ymd(tobacoo_rep$crt_date) %within% before_month, ]



# 构造含RFM变量的数据框
# 训练集上
condition_1 <- before_customer %>% group_by(open_id) %>% summarise(Recency=min(Recency))
Frequency <- before_customer %>% group_by(open_id) %>% summarise(Frequency=n())
condition_2 <- merge(condition_1, Frequency, by = 'open_id')



# 测试集上(最后一个月)
test_tobacoo_1 <- last_customer %>% group_by(open_id) %>% summarise(Recency=min(Recency))
test_Frequency <- last_customer %>% group_by(open_id) %>% summarise(Frequency=n())
test_tobacoo_2 <- merge(test_tobacoo_1, test_Frequency, by = 'open_id')



# 标准化各列数值(训练集)
condition_3 <- condition_2
condition_3$rank_R <- cut(condition_3$Recency, 5, labels = F)
condition_3$rank_R <- 6- condition_3$rank_R   
condition_3$rank_F <- cut(condition_3$Frequency, 5, labels = F)
condition_3$rank_RF <- 0.5*condition_3$rank_R + 0.5*condition_3$rank_F 



# 定义购买与流失(也就是标签列label, 其中设定1-购买，0-流失)
# 前11个月消费者
condition_4 <- merge(condition_3, test_tobacoo_2, by='open_id', all.x = T)
condition_4$label <- ifelse(condition_4$Frequency.y!='NA', 1, 0)
condition_4$label <- ifelse(is.na(condition_4$label), 0, 1)
condition_4 <- condition_4[, -c(7,8)]
names(condition_4)[2:3] <- c('Recency', 'Frequency') # churn: purchase= 0.87:0.13


# 导入district变量(消费者出现消费最多次数的地方)
district <- tobacoo_rep %>% group_by(open_id) %>% summarise(district=max(district))
condition_4 <- merge(condition_4, district, by='open_id', all.x = T)
condition_4[, c(8,7)] <- condition_4[, c(7,8)]
names(condition_4)[8:7] <- names(condition_4[7:8])


### 导入并整合会员数据(多源数据融合，丰富字段)
member_ship <- fread('E:\\project\\会员数据.csv', encoding = 'UTF-8')
member_ship <- member_ship[,-c(2:5, 13:17, 20:22)]

#设置条件，清洗会员数据
cond1 <- member_ship$monthly_income!=""
cond2 <- member_ship$education!=""
cond3 <- member_ship$occupation!=""
member_ship <- member_ship[cond1&cond2&cond3, ]


# 整合消费数据和会员数据
combine_data<- merge(x = condition_4, y = member_ship, by = 'open_id', all.x = T)

# 添加新列event_result
temp_event_result <- tobacoo_rep %>% group_by(open_id) %>% summarise(event_result=max(event_result))
table(temp_event_result$event_result)


# 合并到初始数据
combine_data1 <- merge(combine_data, temp_event_result, by='open_id', all.x = T)
cond4 <-!is.na(combine_data1$monthly_income)
combine_data2 <- combine_data1[cond4, ]


### 调整列顺序
combine_data2[,c(1:18)] <- combine_data2[,c(1, 10:12,9, 13:15,7, 16:18, 2:6, 8)]
names(combine_data2) <- c('open_id','sex','birthday','email','phone_type','monthly_income','education','occupation',
                        'district','total_integral','surplus_integral','event_result','Recency','Frequency','rank_R',
                        'rank_F','rank_RF','label')


# 除去email phone_type等敏感列
combine_data3<- combine_data2[, -c(4,5)]

######将离散变量映射为数值

## 处理(地区)district变量
library(car)
table(combine_data3$district)
combine_data3$district <- as.factor(combine_data3$district)
combine_data3$district <- recode(combine_data3$district, "'李沧区'=1; '崂山区'=2; '市南区'=3; '城阳区'=4; 
                                                          '市北区'=5; '胶州市'=6; '莱西市'=7; '黄岛区'=8;
                                                            '即墨市'=9; '平度市'=10")


## 处理(生日)日期变量
combine_data3$birthday <- combine_data3$birthday %>% ymd()
aggr(combine_data3, number=T, prop=F)
hist(combine_data3$birthday, breaks = 'year')

# 插补缺失值
set.seed(666)
for(i in 1:nrow(combine_data3)){
  if(is.na(combine_data3[i,3])==TRUE){  
    combine_data3[i, 3] <-  as.Date('1975-01-01') + runif(n=1, min=0, max=3600) # 均匀分布插补 
  }
}


# 查看birthday缺失值，并插入
anyNA(combine_data3$birthday)
set.seed(888)
for(i in 1:nrow(combine_data3)){
  if((ymd(combine_data3[i,3])>ymd('2000-12-31'))==TRUE){  # 大于2000年的异常值，重新插补处理
    combine_data3[i,3] <- as.Date('1975-01-01') + runif(n=1, min=0, max=3600)
  }
}

# 再次检验缺失值情况
aggr(combine_data3, number=T, prop=F)
hist(combine_data3$birthday, breaks = 'year')


## 将生日变量转化成年龄,在分段编码数字化
combine_data3$age <- difftime(time1 = '2018-7-20', time2 = combine_data3$birthday, units = 'days') %>% round() %>% as.numeric()
combine_data3$age <- round(combine_data3$age/365)
combine_data3$birthday <- combine_data3$age
names(combine_data3)[3] <- c('age')
combine_data3 <- combine_data3[, -17]


### 处理sex(性别)变量 男-1, 女-0
table(combine_data3$sex)
combine_data3$sex <- as.factor(combine_data3$sex)
combine_data3$sex <- recode(combine_data3$sex, "'男'=1;'女'=2")
table(combine_data3$sex)


##处理monthly_income变量 
#### 3000以下:1[#523], 3000-5000:2[#956], 5000~10000:3[#721], 10000-20000:4[#144], 20000以上:5[#94]
table(combine_data3$monthly_income)
combine_data3$monthly_income <- as.factor(combine_data3$monthly_income)
combine_data3$monthly_income <- recode(combine_data3$monthly_income,"'3000以下'=1;
                                     '3000-5000'=2;
                                     '5000-10000'=3;
                                     '10000-20000'=4;
                                     '20000以上'=5")


###处理education变量
#### 高中以下:1[#925], 中专及高中:2[#826], 大专:3[#447], 本科:4[#190], 研究生以上:5[#50]
table(combine_data3$education)
combine_data3$education <- recode(combine_data3$education, "'高中以下'=1;
                                '中专及高中'=2;
                                '大专'=3;
                                '本科'=4;
                                '研究生以上'=5")



##处理ocupation变量
####退休:1[#19],学生:2[#10],农民:3[#213],其他:4[#406],公务员及事业单位:5[#169],国有企业:6[#221],自由职业:7[#712].私营企业:8[#688],
table(combine_data3$occupation)
combine_data3$occupation <- recode(combine_data3$occupation,"'退休'=1;
                                                        '学生'=2;
                                                        '农民'=3;
                                                        '其他'=4;
                                                        '公务员及事业单位'=5;
                                                        '国有企业'=6;
                                                        '自由职业'=7;
                                                        '私营企业'=8")



##处理event_result变量
#### 红包:0[#123], 青币:1[#2315]
table(combine_data3$event_result)
combine_data3$event_result <- recode(combine_data3$event_result,"'青币'=1; '红包'=2; '京东礼品'=2; '彩票'=2;''=2")
table(combine_data3$event_result)



### 添加diversity 和 loyalty变量(通过对经纬度数据转化得来)
div_loy_result <- read.csv('E\\:project\\originData\\div_loy_result.csv')
div_loy_result$open_id <- as.character(div_loy_result$open_id)
combine_data4 <- merge(combine_data3, div_loy_result,by = 'open_id', all.x = T)
combine_data4[, c(16, 17, 18)] <- combine_data4[, c(17, 18, 16)]
names(combine_data4)[16:18] <- c('diversity', 'loyalty', 'label')


### 标准化diversity数值(loyalty本身就是概率值、无需再归一化处理)
combine_data5 <- combine_data4
combine_data5$diversity <- (combine_data4$diversity-min(combine_data4$diversity))/(max(combine_data4$diversity)-min(combine_data4$diversity))


### 得到了最终所需预处理数据集，下面保存数据下面进入实体嵌入环节。
write.csv(combine_data4, 'E\\:project\\originData\\\\combine_data4.csv')
write.csv(combine_data5, 'E\\:project\\originData\\\\combine_data5.csv')


#########################################################(2)实体嵌入#####################################################################

require(discretization)
require(keras)


# 导入数据
combine_data4 <- read.csv('combine_data4.csv')


### 将各列都数值化(as.numeric)
for(i in 2:ncol(combine_data4)){
  combine_data4[,i] <- as.numeric(combine_data4[,i])
}
str(combine_data4)
prop.table(table(combine_data4$label))



### chiM处理不同量纲的变量(combine_data4)
train <- combine_data4[,-c(1,2,4:6)]
train <- data.frame(train)
chi <- chiM(train,alpha = 0.05)
disc_data <- chi$Disc.data
View(disc_data)
combine_data4[,c(3,7:18)] <- disc_data[,1:13]

# 将处理后的数据复制给run_data2, 再进行后续实验，防止实验数据出错反工。
run_data2 <- combine_data4
run_data2 <- run_data2[,-1]
dim(run_data2)


# 运用keras框架进行实体嵌入处理run_data2中部分变量
model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$sex)+1, output_dim = 10, mask_zero = FALSE)
prediction1 <- model %>% predict(run_data2$sex)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$age)+1, output_dim = 10, mask_zero = FALSE)
prediction2 <- model %>% predict(run_data2$age)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$monthly_income)+1, output_dim = 10, mask_zero = FALSE)
prediction3 <- model %>% predict(run_data2$monthly_income)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$education)+1, output_dim = 10, mask_zero = FALSE)
prediction4 <- model %>% predict(run_data2$education)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$occupation)+1, output_dim = 10, mask_zero = FALSE)
prediction5 <- model %>% predict(run_data2$occupation)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$district)+1, output_dim = 10, mask_zero = FALSE)
prediction6 <- model %>% predict(run_data2$district)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$total_integral)+1, output_dim = 10, mask_zero = FALSE)
prediction7 <- model %>% predict(run_data2$total_integral)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$surplus_integral)+1, output_dim = 10, mask_zero = FALSE)
prediction8 <- model %>% predict(run_data2$surplus_integral)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$event_result)+1, output_dim = 10, mask_zero = FALSE)
prediction9 <- model %>% predict(run_data2$event_result)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$Recency)+1, output_dim = 10, mask_zero = FALSE)
prediction10 <- model %>% predict(run_data2$Recency)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$Frequency)+1, output_dim = 10, mask_zero = FALSE)
prediction11 <- model %>% predict(run_data2$Frequency)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$rank_R)+1, output_dim = 10, mask_zero = FALSE)
prediction12 <- model %>% predict(run_data2$rank_R)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$rank_F)+1, output_dim = 10, mask_zero = FALSE)
prediction13 <- model %>% predict(run_data2$rank_F)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$rank_RF)+1, output_dim = 10, mask_zero = FALSE)
prediction14 <- model %>% predict(run_data2$rank_RF)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$diversity)+1, output_dim = 10, mask_zero = FALSE)
prediction15 <- model %>% predict(run_data2$diversity)

model <- keras_model_sequential()
model %>% layer_embedding(input_dim = max(run_data2$loyalty)+1, output_dim = 10, mask_zero = FALSE)
prediction16 <- model %>% predict(run_data2$loyalty)

result2 <- cbind(data.frame(prediction1,prediction2,prediction3,prediction4,prediction5,prediction6,prediction7,prediction8,prediction9,prediction10,
                           prediction11, prediction12, prediction13, prediction14, prediction15, prediction16))


try2 <- cbind(result2, run_data2$label)
names(try2)[161] <- c('label')
View(try2)



#########################################################(3)深度学习/机器学习实证预测#######################################################


#为实验结果度量构建列表
auc=list()
acc=list()
tmp=list()
emx=list()

## 深度学习模型
##自己构建的EE-CNN model (取各模型10次结果平均值作为最终结果)
for (i in 1:10){
  set.seed(111)
  require(caret)
  inTASK <- createDataPartition(y=try2$label,p=0.7,list=FALSE)
  training <- try2[inTASK,]
  validate <- try2[-inTASK,]
  training <- data.matrix(training)
  validate <- data.matrix(validate)
  training.x <- training[,-161]
  training.y <- training[,161]
  validate.x <- validate[,-161]
  validate.y <- validate[,161]
  
  img_rows <- 10
  img_cols <- 16
  num_classes <- 3
  dim(training.x) <- c(nrow(training.x), img_rows, img_cols, 1)
  dim(validate.x) <- c(nrow(validate.x), img_rows, img_cols, 1)
  input_shape <- c(img_rows, img_cols, 1)
  require(keras)
  training.y <- to_categorical(training.y, num_classes)
  validate.y <- to_categorical(validate.y, num_classes)
  training.y <- training.y[,-3]
  validate.y <- validate.y[,-3]
  num_classes <- 2
  
  model <- keras_model_sequential()
  
  model %>%
    layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                  input_shape = input_shape) %>%
    layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_flatten() %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate=0.5)%>%
    layer_dense(units = num_classes, activation = 'softmax')
  
  model %>% compile(
    loss = loss_binary_crossentropy,
    optimizer = optimizer_adam(),
    metrics = c('accuracy')
  )
  epochs <- 15
  batch_size <- 64
  im_history <- model %>% fit(
    training.x, training.y,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 1,
    validation_data = list(validate.x, validate.y))
  
  preds <- predict(model,validate.x)
  require(pROC)
  label <- factor(data.frame(validate.y)[,2])
  score <- data.frame(preds)[,2]
  ROC <- roc(label,score)
  pred.label <- max.col((preds)) - 1
  misClasificError <- mean(pred.label != label)
  require(EMP)
  em <- empCreditScoring(score,label)
  require(ROCR)
  df <- prediction(score,label)
  perf <- performance(df,"tpr","fpr")
  # auc <- performance(df,"auc")
  acc[[i]] <- 1-misClasificError
  auc[[i]] <- ROC$auc
  tmp[[i]] <-max(attr(perf,"y.values")[[1]]-attr(perf,"x.values")[[1]])
  emx[[i]] <-em$EMPC 
}





####一般机器学习模型
train2 <- data.frame(run_data2)


##Logistic Regression模型
auc1=list()
acc1=list()
tmp1=list()
em1=list()

for (i in 1:10){         
  set.seed(222)
  require(caret)
  inTASK <- createDataPartition(y=train2$label,p=0.7,list=FALSE)
  training <- train2[inTASK,]
  validate <- train2[-inTASK,]
  
  model1 <- glm(label~.,family=binomial(link='logit'),data=training)
  model1 <- step(model1)
  pre1 <- predict(model1,type = "response",newdata=validate) # pre1就只有一列，都是0-1之间的概率值
  require(EMP)
  score <- data.frame(pre1)[,1]
  label <- factor(data.frame(validate)[,17])
  em <- empCreditScoring(score,label)
  require(ROCR)
  x1 <- prediction(pre1,validate$label)
  pre1 <- ifelse(pre1> 0.5,1,0)
  misClasificError <- mean(pre1 != validate$label)
  perf1 <- performance(x1,"tpr","fpr")
  
  auc.tmp1 <- performance(x1,"auc")
  acc1[[i]] <- 1-misClasificError
  auc1[[i]] <- as.numeric(auc.tmp1@y.values)  # @符号是代表?
  tmp1[[i]]<-max(attr(perf1,"y.values")[[1]]-attr(perf1,"x.values")[[1]])
  em1[[i]] <- em$EMPC
}



#Random Forest模型               
require(randomForest)
auc2=list()
acc2=list()
tmp2=list()
em2=list()

for (i in 1:10){
  set.seed(1234)
  require(caret)
  inTASK <- createDataPartition(y=train2$label,p=0.7,list=FALSE)
  training <- train2[inTASK,]
  validate <- train2[-inTASK,]
  training$label <- as.factor(training$label)
  fit.forest <- randomForest(label~.,data = training,na.action= na.roughfix, importance = TRUE) #na.action= na.roughfix
  forest.pred <- predict(fit.forest, validate)
  require(EMP)
  score <- data.frame(forest.pred)[,1] 
  label <- factor(data.frame(validate)[,17])
  em <- empCreditScoring(as.numeric(score),as.numeric(label))
  require(ROCR)
  # x2 <- prediction(as.numeric(forest.pred),as.numeric(validate$label))
  x2 <- prediction(as.numeric(score),as.numeric(label))
  # predict.results <- ifelse(as.numeric(forest.pred)>0.5, 1, 0)
  predict.results <- forest.pred
  misClasificError <- mean(predict.results != validate$label) 
  perf2 <- performance(x2,"tpr","fpr")
  auc.tmp2 <- performance(x2,"auc")
  acc2[[i]] <- 1-misClasificError
  auc2[[i]] <- as.numeric(auc.tmp2@y.values)
  tmp2[[i]]<-max(attr(perf2,"y.values")[[1]]-attr(perf2,"x.values")[[1]])
  em2[[i]] <- em$EMPC
} 



##SVM模型
acc3=list()
auc3=list()
tmp3=list()
em3=list()

for (i in 1:10){
  set.seed(444)
  require(e1071)
  require(caret)
  inTASK <- createDataPartition(y=train2$label,p=0.7,list=FALSE)
  training <- train2[inTASK,]
  validate <- train2[-inTASK,]
  
  model1 <- svm(label~.,data=training)
  pre1 <- predict(model1,validate)
  require(EMP)
  score <- data.frame(pre1)[,1]
  label <- factor(data.frame(validate)[,17])
  em <- empCreditScoring(score,label)
  ROC <- roc(label,score)
  require(ROCR)
  x1 <- prediction(pre1,validate$label)
  perf1 <- performance(x1,"tpr","fpr")
  pre1 <- factor(ifelse(pre1> 0.5,1,0),levels = c('0','1'))
  misClasificError <- mean(pre1 != validate$label)
  acc3[[i]]  <- 1-misClasificError
  auc3[[i]]  <- ROC$auc
  tmp3[[i]] <-max(attr(perf1,"y.values")[[1]]-attr(perf1,"x.values")[[1]])
  em3[[i]] <- em$EMPC
}












































































































































