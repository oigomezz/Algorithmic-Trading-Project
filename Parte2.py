#########################################################################################
# We use a Gaussian Naive Bayes model to predict if a stock will have a high return 
# or low return next Monday (num_holding_days = 5),  using as input decision variables 
#  the assets growthto yesterday from 2,3,,4,5,6,7,8,9 and 10 days before  
#########################################################################################

##################################################
# Imports
##################################################

# Pipeline and Quantopian Trading Functions
import quantopian.algorithm as algo
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline, CustomFactor 
from quantopian.pipeline.factors import Returns
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.data import Fundamentals

# The basics
from collections import OrderedDict
import time
import pandas as pd
import numpy as np

# SKLearn :)
from sklearn.naive_bayes import GaussianNB

##################################################
# Globals
##################################################

num_holding_days = 5 # holding our stocks for five trading days.
days_for_fundamentals_analysis = 60
upper_percentile = 90
lower_percentile = 100 - upper_percentile
MAX_GROSS_EXPOSURE = 1.0
MAX_POSITION_CONCENTRATION = 0.01

##################################################
# Initialize
##################################################

def initialize(context):
    """ Called once at the start of the algorithm. """

    # Configure the setup
#    set_commission(commission.PerShare(cost=0.001, min_trade_cost=0))
#    set_asset_restrictions(security_lists.restrict_leveraged_etfs)

    # Schedule our function
    algo.schedule_function(
        rebalance,
        algo.date_rules.week_start(days_offset=0),
        algo.time_rules.market_open() 
    )

    # Build the Pipeline
    algo.attach_pipeline(make_pipeline(context), 'my_pipeline')

##################################################
# Pipeline-Related Code
##################################################
            
class Predictor(CustomFactor):
    """ Defines our machine learning model. """
    
    # The factors that we want to pass to the compute function. We use an ordered dict for clear labeling of our inputs.
    factor_dict = OrderedDict([
              ('Asset_Growth_2d' , Returns(window_length=2)),
              ('Asset_Growth_3d' , Returns(window_length=3)),
              ('Asset_Growth_4d' , Returns(window_length=4)),
              ('Asset_Growth_5d' , Returns(window_length=5)),
              ('Asset_Growth_6d' , Returns(window_length=6)),
              ('Asset_Growth_7d' , Returns(window_length=7)),
              ('Asset_Growth_8d' , Returns(window_length=8)),
              ('Asset_Growth_9d' , Returns(window_length=9)),
              ('Asset_Growth_10d' , Returns(window_length=10)),
              ('Return' , Returns(inputs=[USEquityPricing.open],window_length=5))
              ])

    columns = factor_dict.keys()
    inputs = factor_dict.values()

    # Run it.
    def compute(self, today, assets, out, *inputs):
        """ Through trial and error, I determined that each item in the input array comes in with rows as days and securities as columns. Most recent data is at the "-1" index. Oldest is at 0.
        !!Note!! In the below code, I'm making the somewhat peculiar choice  of "stacking" the data... you don't have to do that... it's just a design choice... in most cases you'll probably implement this without stacking the data.
        """

        ## Import Data and define y.
        inputs = OrderedDict([(self.columns[i] , pd.DataFrame(inputs[i]).fillna(0,axis=1).fillna(0,axis=1)) for i in range(len(inputs))]) # bring in data with some null handling.
        num_secs = len(inputs['Return'].columns)
        y = inputs['Return'].shift(-num_holding_days-1)
        
        for index, row in y.iterrows():
            
             upper = np.nanpercentile(row, upper_percentile)
             lower = np.nanpercentile(row, lower_percentile)
             upper_mask = (row >= upper)
             lower_mask = (row <= lower)          
             row = np.zeros_like(row)
             row[upper_mask]= 1
             row[lower_mask]=-1
             y.iloc[index] = row
            
        y=y.stack(dropna=False)
        
        
        ## Get rid of our y value as an input into our machine learning algorithm.
        del inputs['Return']

        ## Munge x and y
        x = pd.concat([df.stack(dropna=False) for df in inputs.values()], axis=1).fillna(0)
        
        ## Run Model
        model = GaussianNB() 
        model_x = x[:-num_secs*(num_holding_days+1)]
        model_y = y[:-num_secs*(num_holding_days+1)]
        model.fit(model_x, model_y)

        out[:] = model.predict_proba(x[-num_secs:])[:, 1]

def make_pipeline(context):

    universe = QTradableStocksUS()
      
    predictions = Predictor(window_length=days_for_fundamentals_analysis, mask=universe) #mask=universe
    
       
    low_future_returns = predictions.percentile_between(0,lower_percentile)
    high_future_returns = predictions.percentile_between(upper_percentile,100)
   
    securities_to_trade = (low_future_returns | high_future_returns)
    pipe = Pipeline(
        columns={
            'predictions': predictions
        },
        screen=securities_to_trade
    )

    return pipe

def before_trading_start(context, data):
      
    context.output = algo.pipeline_output('my_pipeline')

    context.predictions = context.output['predictions']

##################################################
# Execution Functions
##################################################

def rebalance(context,data):
    # Timeit!
    start_time = time.time()
    
    objective = opt.MaximizeAlpha(context.predictions)
    
    max_gross_exposure = opt.MaxGrossExposure(MAX_GROSS_EXPOSURE)
    
    max_position_concentration = opt.PositionConcentration.with_equal_bounds(
        -MAX_POSITION_CONCENTRATION,
        MAX_POSITION_CONCENTRATION
    )
    
    dollar_neutral = opt.DollarNeutral()
    
    constraints = [
        max_gross_exposure,
        max_position_concentration,
        dollar_neutral,
    ]

    algo.order_optimal_portfolio(objective, constraints)

    # Print useful things. You could also track these with the "record" function.
    print 'Full Rebalance Computed Seconds: '+'{0:.2f}'.format(time.time() - start_time)
    print "Leverage: " + str(context.account.leverage)
