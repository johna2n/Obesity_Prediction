# install.packages("remotes")
# remotes::install_github("privefl/bigsnpr")
library(bigstatsr)
library(bigsnpr)
library(caret)
library(gam)

set.seed(1)
#setwd('/home/nhanta/PRS/covid19')
rds <- snp_readBed("work/target.QC.bed")
bed <- snp_attach(rds)
bed_matrix <- bed$genotypes
na.gam.replace(bed_matrix)

split <- sample(c(rep(0, 0.8 * nrow(bed_matrix)),  # Create dummy for splitting
                        rep(1, 0.2 * nrow(bed_matrix))))
# Load data
training_data = bed_matrix[split == 0,]
testing_data = bed_matrix[split == 1,]

# Load covariate
covariate = read.table('work/target.covariate', header = TRUE)
covariate = covariate[, 3:ncol(covariate)]

covar_train = covariate[split == 0, ]
covar_test = covariate[split == 1, ]

covar_train <- as.matrix(covar_train)
covar_test <- as.matrix(covar_test)

# Convert data to big matrix
N <- dim(training_data)[1]
M <- dim(training_data)[2]
X_train <- FBM(N, M, init = training_data)

n <- dim(testing_data)[1]
m <- dim(testing_data)[2]
X_test <- FBM(n, m, init = testing_data)

# Convert output to binary
pheno <- read.table('work/target.phenotype', header = TRUE)
pheno <- pheno[,3]
y_train <- pheno[split==0]
y_train <- unlist(y_train, use.names = FALSE)

y_test <- pheno[split==1]
y_test <- unlist(y_test, use.names = FALSE)

get_binary <-  function(data){
  as = c()
  for (x in data){
    if (x ==1){
      as = append(as, 0)
    }else{
      as = append(as, 1)
    }
  }
return(as)
}

y_train <- get_binary(y_train)
y_test <- get_binary(y_test)

start_time <- Sys.time()
mod <- big_spLogReg(X_train, y_train, covar.train = covar_train, alphas = c(1, 0.1), K = 5, warn = FALSE, ncores = 10)

#summary(mod)
end_time <- Sys.time()
print(end_time - start_time)

pred <- predict(mod, X_test, covar.row = covar_test)
summary(mod, best.only = TRUE)
AUC(pred, y_test)
write.csv(summary(mod, best.only = TRUE)$beta, 'work/beta.csv')
id = attr(mod, "ind.col")
write.csv(id, 'work/id.csv')
quit() # Exist R
