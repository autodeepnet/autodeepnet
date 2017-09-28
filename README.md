# auto_deepnet
Automated Deep Learning with Neural Networks

## Project Goals
Automate deep learning for common tasks! Much as [auto_ml](http://auto.ml) automates machine learning at production quality, this package automates deep learning for production, using a set of best practices that propogate through the latest advances from leading AI labs.

### Why?
There are a bunch of different reasons why, depending on who you are

#### AI Expert
- Automate the boring and repetitive parts of the deep learning process
- Lets you finish the less exciting projects more rapidly, while freeing up time for the complex projects that require more custom work
- Faster iteration speed
- Reduces the amount of code that has to go through code review with each new model you've shipped
- Many minor best practices baked in- the kinds of things that aren't too difficult to do, but are just annoying to do for each and every new project
- Still get to customize and control most of the important components under the hood by passing in your own config values

#### Engineer
- Approach deep learning like any other open source engineering project- read the docs, interact with the API we've laid out, and don't worry about what's going on under the hood unles you really want to
- Code is tested and optimized to run at production speeds- significantly reduces the space for bugs
- A reasonable set of default values is already provided for everything- you only have to tweak settings if you want to
- Have more business impact more rapidly
- See if deep learning is useful for your project or not without investing too much time

#### Newbie
- You only need a few lines of Python code to get production-ready deep learning up and running
- See what all the buzzwords are about, and see if it's actually impactful for your area or not (hint- anything that's overhyped is unlikely to live up to all the expectations people are putting on it)
- We handle most of the confusing parts- all you gotta do is give us the dataset

## Project Status- Work in Progress
This project is under active development by a team of engineers. It's designed to be modular, so if you want to jump in and contribute, we'd love to have you!


## Main Workstreams
- Independent data processing pipelines for different data types (images, audio, time series forecasting, tabular data, etc.)
- Independent hyperparameter definition and model creation scripts for each realm
- One central model training babysitter- give it a compiled deep learning model, and it handles the babysitting while the model trains
- One central hyperparameter optimization package- Given that we're only optimizing deep learning, we can do things that don't generalize to other machine learning models, like increase the dropout rate when we encounter overfitting

Again, if you want to contribute, open an issue and get in touch!


