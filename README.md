# EARL Program Optimisation



Next week will be the first EARL conference in San Francisco.. The team here at Mango have worked hard to put together an excellent program with great speakers. However with so many interesting talks it can be hard to decide which to attend. For example, Rich Pugh is a huge fan of Shiny and would like to attend all talks involving Shiny. Of course we can put together a personal schedule for Rich by hand but according to Rich it's "easier" to let R do it for you. 

First, we converted the [program](https://earlconf.com/downloads/sanfrancisco/EARL-SF-2017-Agenda.pdf) into a tabular format. There are two streams, twelve sessions and eighteen time slots:


```r
library(readr)
program <- read_csv("program.csv")
head(program)
## # A tibble: 6 × 6
##    Time                                     Author
##   <int>                                      <chr>
## 1     1                    Joseph Rickert, RStudio
## 2     2             Gabriela de Queiroz, SelfScore
## 3     3              Mark Sellors, Mango Solutions
## 4     4                       Szilard Pafka, Epoch
## 5     5    David Bishop, Hitachi Solutions America
## 6     6 Dr Aleksander Dietrichson, Blackboard Inc.
## # ... with 4 more variables: Title <chr>, Stream <int>, Session <int>,
## #   AbstractID <int>
```

### Because I'm Shiny!

To be able to optimise our personal schedule for maximum "shinyness" we must first calculate the "shinyness" of each talk. In a previous [blogpost](http://www.mango-solutions.com/wp/2017/03/writing-a-conference-abstract-the-data-science-way/) we showed how to read in the abstracts of all presentations and we have done the same here. We then calculate the TF-IDF score of the word "Shiny" for each abstract as our measure for "shinyness".


```r
abstracts <- read_csv("abstracts.csv")
source("Scripts/extract_abstracts.R")
library(tidytext)
data(stop_words)

tidy_abstracts <- abstracts %>% 
  mutate(Abstract=remove_punctuation(Abstract)) %>% 
  unnest_tokens(word, Abstract) %>% # abracadabra!
  anti_join(stop_words %>% filter(word!="r")) %>% # In this case R is a word
  filter(is.na(as.numeric(word))) # filter out numbers

# calculate shinyness as the TF-IDF score of the word "shiny"
shinyness <- tidy_abstracts %>% 
  count(AbstractID, word, sort=TRUE) %>% 
  ungroup() %>% 
  bind_tf_idf(word, AbstractID, n) %>% 
  filter(word=="shiny") %>% 
  select(AbstractID, tf_idf)
# TF-IDF might be overkill in this case as a simple term frequency would probably give the same result

# add column shinyness to program
program <- program %>% 
  left_join(shinyness) %>% 
  rename(shinyness = tf_idf) %>% 
  # give talks without shinyness a default score
  mutate(shinyness = ifelse(is.na(shinyness), 1e-3, shinyness)) 
```


### The Optimisation

This a integer programming problem as our decision variables are binary, we either attend a talk or we don't. Our mathematical formulation is as follows:

$$ \max \sum_{i=1}^{N}\sum_{j=1}^{M} s_{ij}*x_{ij} \\ s.t. \space \sum_{i}^{N} x_{ij} \leq 1 \space\space \forall j$$

Our decision variable is $x_{ij}$ which is 1 if we are attending talk in stream $i$ at time $j$ and 0 otherwise. The $s_{ij}$ stand for the shinyness of each talk. Finally, the constraint states we cannot attend two talks at the same time ^[This optimisation is not built for Jeff Leek. Jeff Leek always attends all talks].

To implement this in R we will use the **lpSolveAPI** package.


```r
library(lpSolveAPI)
# number of decision variables = number of talks
numVariables <- nrow(program)
# number of contraints = number of time slots
numConstraints <- length(unique(program$Time))

# add talk id to program
program$ID <- 1:numVariables

# instantiate the model
lpModel <- make.lp(numConstraints, numVariables)
# set the coefficients on each decision variable in each constraint
for(j in 1:nrow(program)){
  set.column(lpModel, j, 1, program[program$ID==j, "Time"])
}

# set the right hand side of each constraint
set.constr.value(lpModel, rhs=rep(1, numConstraints), constraints=1:numConstraints)
# set the objective value (the shinyness of each talk)
set.objfn(lpModel, program$shinyness)
# set the type of decision variable
set.type(lpModel, 1:numVariables, type="binary")
# set the optimisation direction (maximise)
cntrl <- lp.control(lpModel, sense="max")

# write to file to inspect model
# write.lp(lpModel,'model.lp',type='lp')

# solve the model
result <- solve(lpModel)
solution <- get.variables(lpModel)
```

The `solve` function returns a status code which is zero if a solution is found. More information can be found in the help file of the function. The `get.variables` function returns the value of our decision variable in the optimal solution. We can use these values to find out which talks to attend.


```r
program %>% filter(solution==1) %>% 
  select(Time, Stream, Session, Title) %>% 
  mutate(Title=paste(substring(Title, 0, 30), "...")) %>% 
  arrange(Time) %>% 
  select(Time, Title)
```

```
## # A tibble: 18 × 2
##     Time                              Title
##    <int>                              <chr>
## 1      1 the missing manual for running ...
## 2      2 using data to identify risky p ...
## 3      3 how we built a shiny app for 7 ...
## 4      4 crimes and data predictive pol ...
## 5      5 r as a pharmacokinetic simulat ...
## 6      6 going enterprise with xray lea ...
## 7      7 using shiny and r to build pow ...
## 8      8       treating patients with r ...
## 9      9 predicting patient lengthofsta ...
## 10    10 staying on top of the beat how ...
## 11    11 shiny as a replacement for sas ...
## 12    12 largescale reproducible simula ...
## 13    13 understanding server traffic a ...
## 14    14 beyond popularity monetizing r ...
## 15    15       oil through the pipeline ...
## 16    16 the pharmaceutical industrys p ...
## 17    17 pioneering r for clinical deve ...
## 18    18 powering apis with r and build ...
```

There it is, a personal schedule optimised for maximum shinyness. You can of course tweak the code to create a personal schedule optimised for your personal preferences. Either way, we hope you enjoy a wonderful conference.
