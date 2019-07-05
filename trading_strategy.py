# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 11:07:53 2018

@author: Administrator
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
def equity_curve(predict,initial_money=1000000, slippage=1/1000, commission_rate=1/1000,
                 trading_units=1, margin_ratio=0.1, stop_loss=1,Rf=0.035):
    # :param initial_money: 初始资金，默认为1000000元（假设了份额数可分）
    # :param slippage: 滑点，默认为万分之2
    # :param commission_rate: 手续费，commission fees，默认为万分之2
    # :param trading_units: 交易单位，默认为1
    # :param margin_ratio: 保证金率，默认为10%
    # :param Rf: 无风险b收益率，默认为3.5%
    
    # 交易逻辑：首先判断是否需要改变仓位。（1）若不需要，分为多头、空头以及空仓；（2）若需要，按当天状态分为多头空头以及空仓三类。
    # 当天是多头时：（1）空头变多头 （2）空仓变多头。当天是空仓时：（1）空头变空仓 （2）多头变空仓。当天是空头时：（1）空仓变空头
    #（2）多头变空头。最后一天，平多头和平空头。
    #建仓和平仓时，需计算手续费，静态权益改变。静态权益：计算建仓和平仓时的损益
    
   
    # ===第一天的情况
    open_pos_num = 0  # 建仓价格序号
    close_pos_num = 0  # 平仓价格序号
    ret = predict.copy()
    begin = ret.index[0]
    ret.at[begin, 'hold_num'] = 0  # 持有份额数
    ret.at[begin, 'short_margin'] = np.nan  # 空头保证金
    ret.at[begin, 'long_margin'] = np.nan  # 多头保证金
    ret.at[begin, 'entry_price'] = np.nan  # 买入价格
    ret.at[begin, 'out_price'] = np.nan  # 卖出价格
    ret.at[begin, 'cash'] = initial_money  # 持有现金
    ret.at[begin, 'static_equity'] = initial_money  # 静态权益
    ret.at[begin, 'dynamic_equity'] = initial_money  # 动态权益
    trade_open = pd.DataFrame()  # 记录建仓时交易数据
    trade_close = pd.DataFrame()  # 记录平仓交易数据
    
    # # ===第一天之后每天的情况
    for i in ret.index[1:]:
        # 判断是否需要调整仓位
        # 不需要调整仓位
        # print(i)
        if ret.at[i,'pos'] == ret.at[i - 1,'pos']:
            ret.at[i,'hold_num'] = ret.at[i - 1,'hold_num']
            ret.at[i, 'static_equity'] = ret.at[i-1, 'static_equity']
            if ret.at[i,'pos'] == 0:
                # 空仓
                ret.at[i, 'short_margin'] = 0
                ret.at[i, 'long_margin'] = 0
                # 空仓时：动态权益 = 静态权益
                ret.at[i, 'dynamic_equity'] = ret.at[i, 'static_equity']
                ret.at[i, 'cash'] = ret.at[i, 'dynamic_equity']
            if ret.at[i,'pos'] == 1:
                # 多头
                ret.at[i, 'short_margin'] = 0
                # 占用保证金：收盘价 * 保证金比率
                ret.at[i, 'long_margin'] = ret.at[i,'hold_num']*ret.at[i,'close']*margin_ratio 
                # 持多头：动态权益 = 静态权益 + (收盘价 - 建仓价) * 持有份额
                ret.at[i, 'dynamic_equity'] = ret.at[i, 'static_equity'] + (ret.at[i,'close']-trade_open.at[open_pos_num,'open_pos_price'])*ret.at[i,'hold_num']
                # 可用资金：动态权益 - 占用保证金
                ret.at[i, 'cash'] = ret.at[i, 'dynamic_equity'] - ret.at[i, 'long_margin']
            if ret.at[i,'pos'] == -1:
                # 空头
                ret.at[i, 'short_margin'] = ret.at[i,'hold_num']*ret.at[i,'close']*margin_ratio
                ret.at[i, 'long_margin'] = 0
                # 持空头：动态权益 = 静态权益 + (建仓价 - 收盘价) * 持有份额
                ret.at[i, 'dynamic_equity'] = ret.at[i, 'static_equity'] + (trade_open.at[open_pos_num,'open_pos_price'] - ret.at[i,'close'])*ret.at[i,'hold_num']
                ret.at[i, 'cash'] = ret.at[i, 'dynamic_equity'] - ret.at[i, 'short_margin']
        if ret.at[i,'pos'] != ret.at[i - 1,'pos']:
            # 当天是多头时
            if ret.at[i,'pos'] == 1:
                #空头变多头
                if  ret.at[i-1,'pos'] == -1:
                    ## 空头先平仓
                    ret.at[i, 'out_price'] = ret.at[i,'open']*(1+slippage)
                    ret.at[i, 'hold_num'] = 0
                    close_pos_num+=1
                    trade_close.at[close_pos_num, 'close_pos_price'] = ret.at[i, 'out_price']
                    trade_close.at[close_pos_num, 'close_date'] = ret.at[i, 'date']
                    ret.at[i, 'commission_cost'] = ret.at[i-1,'hold_num']*ret.at[i, 'out_price']*commission_rate
                    # 平空头：静态权益 = 上一日静态权益 + (建仓价格 -  平仓价格）* 交易份额 - 买卖成本
                    ret.at[i, 'static_equity'] = ret.at[i-1, 'static_equity']+(trade_open.at[open_pos_num,'open_pos_price'] -trade_close.at[close_pos_num, 'close_pos_price'])*ret.at[i-1,'hold_num']- ret.at[i, 'commission_cost']
                    ret.at[i, 'dynamic_equity'] = ret.at[i, 'static_equity'] 
                    ret.at[i, 'cash'] = ret.at[i, 'dynamic_equity']
                    ## 空仓开多头
                    open_pos_num +=1
                    ret.at[i, 'entry_price'] = ret.at[i,'open']*(1+slippage) 
                    ret.at[i, 'hold_num'] = int(ret.at[i, 'cash']/ret.at[i, 'entry_price'])
                    ret.at[i, 'hold_num'] = int(ret.at[i, 'hold_num']/trading_units)*trading_units
                    trade_open.at[open_pos_num, 'open_pos_price'] = ret.at[i, 'entry_price']
                    trade_open.at[open_pos_num, 'open_date'] = ret.at[i, 'date']
                    trade_open.at[open_pos_num, 'type'] = 1
                    trade_open.at[open_pos_num, 'hold_num'] = ret.at[i, 'hold_num']
                    ret.at[i, 'commission_cost'] = ret.at[i, 'commission_cost'] + ret.at[i, 'entry_price']*ret.at[i, 'hold_num']*commission_rate #手续费累加
                    # 静态权益 = 平空头后静态权益 - 建仓手续费
                    ret.at[i, 'static_equity'] = ret.at[i, 'static_equity']-ret.at[i, 'entry_price']*ret.at[i, 'hold_num']*commission_rate # 注意是减累加前的
                    ret.at[i, 'dynamic_equity'] = ret.at[i, 'static_equity']+(ret.at[i,'close']-trade_open.at[open_pos_num,'open_pos_price'])*ret.at[i, 'hold_num']
                if  ret.at[i-1,'pos'] == 0:
                    # 空仓变多头
                    open_pos_num +=1
                    ret.at[i, 'entry_price'] = ret.at[i,'open']*(1+slippage)
                    ret.at[i, 'hold_num'] = int(ret.at[i-1, 'cash']/ret.at[i, 'entry_price'])
                    ret.at[i, 'hold_num'] = int(ret.at[i, 'hold_num']/trading_units)*trading_units
                    trade_open.at[open_pos_num, 'open_pos_price'] = ret.at[i, 'entry_price']
                    trade_open.at[open_pos_num, 'open_date'] = ret.at[i, 'date']
                    trade_open.at[open_pos_num, 'type'] = 1
                    trade_open.at[open_pos_num, 'hold_num'] = ret.at[i, 'hold_num']
                    ret.at[i, 'commission_cost'] = ret.at[i, 'entry_price']*ret.at[i, 'hold_num']*commission_rate
                    ret.at[i, 'static_equity'] = ret.at[i-1, 'static_equity'] - ret.at[i, 'commission_cost']
                    ret.at[i, 'dynamic_equity'] = ret.at[i, 'static_equity']+(ret.at[i,'close']-trade_open.at[open_pos_num,'open_pos_price'])*ret.at[i, 'hold_num']
                ret.at[i, 'short_margin'] = 0
                ret.at[i, 'long_margin'] = ret.at[i,'hold_num']*ret.at[i,'close']*margin_ratio
                ret.at[i, 'cash'] = ret.at[i, 'dynamic_equity'] - ret.at[i, 'long_margin']
                
            # 当天是空头时
            if ret.at[i,'pos'] == -1:
                # 空仓变空头
                if ret.at[i-1,'pos'] == 0:
                    open_pos_num +=1
                    ret.at[i, 'entry_price'] = ret.at[i,'open']*(1-slippage)
                    ret.at[i, 'hold_num'] = int(ret.at[i-1, 'cash']/ret.at[i, 'entry_price'])
                    ret.at[i, 'hold_num'] = int(ret.at[i, 'hold_num']/trading_units)*trading_units
                    trade_open.at[open_pos_num, 'open_pos_price'] = ret.at[i, 'entry_price']
                    trade_open.at[open_pos_num, 'open_date'] = ret.at[i, 'date']
                    trade_open.at[open_pos_num, 'type'] = -1
                    trade_open.at[open_pos_num, 'hold_num'] = ret.at[i, 'hold_num']
                    ret.at[i, 'commission_cost'] = ret.at[i, 'entry_price']*ret.at[i, 'hold_num']*commission_rate
                    ret.at[i, 'static_equity'] = ret.at[i-1, 'static_equity'] - ret.at[i, 'commission_cost']
                    ret.at[i, 'dynamic_equity'] = ret.at[i, 'static_equity']+(trade_open.at[open_pos_num,'open_pos_price']-ret.at[i,'close'])*ret.at[i, 'hold_num']
                if ret.at[i-1,'pos'] == 1:
                    # 多头平仓
                    close_pos_num+=1
                    ret.at[i, 'out_price'] = ret.at[i,'open']*(1-slippage)
                    ret.at[i, 'hold_num'] = 0
                    ret.at[i, 'commission_cost'] = ret.at[i-1,'hold_num']*ret.at[i, 'out_price']*commission_rate
                    trade_close.at[close_pos_num, 'close_pos_price'] = ret.at[i, 'out_price']
                    trade_close.at[close_pos_num, 'close_date'] = ret.at[i, 'date']
                    ret.at[i, 'static_equity'] = ret.at[i-1, 'static_equity']+(trade_close.at[close_pos_num, 'close_pos_price'] - trade_open.at[open_pos_num,'open_pos_price'])*ret.at[i-1,'hold_num']- ret.at[i, 'commission_cost']
                    ret.at[i, 'dynamic_equity'] = ret.at[i, 'static_equity'] 
                    ret.at[i, 'cash'] = ret.at[i, 'dynamic_equity']
                    # 空仓建空头
                    open_pos_num +=1
                    ret.at[i, 'entry_price'] = ret.at[i,'open']*(1-slippage)
                    ret.at[i, 'hold_num'] = int(ret.at[i, 'cash']/ret.at[i, 'entry_price'])
                    ret.at[i, 'hold_num'] = int(ret.at[i, 'hold_num']/trading_units)*trading_units
                    trade_open.at[open_pos_num, 'open_pos_price'] = ret.at[i, 'entry_price']
                    trade_open.at[open_pos_num, 'open_date'] = ret.at[i, 'date']
                    trade_open.at[open_pos_num, 'type'] = -1
                    trade_open.at[open_pos_num, 'hold_num'] = ret.at[i, 'hold_num']
                    ret.at[i, 'commission_cost'] = ret.at[i, 'commission_cost'] + ret.at[i, 'entry_price']*ret.at[i, 'hold_num']*commission_rate
                    ret.at[i, 'static_equity'] = ret.at[i, 'static_equity'] - ret.at[i, 'entry_price']*ret.at[i, 'hold_num']*commission_rate
                    ret.at[i, 'dynamic_equity'] = ret.at[i, 'static_equity']+(trade_open.at[open_pos_num,'open_pos_price']-ret.at[i,'close'])*ret.at[i, 'hold_num']
                ret.at[i, 'short_margin'] = ret.at[i,'hold_num']*ret.at[i,'close']*margin_ratio
                ret.at[i, 'long_margin'] = 0
                ret.at[i, 'cash'] = ret.at[i, 'dynamic_equity'] - ret.at[i, 'short_margin']
                
            # 当天是空仓时
            if ret.at[i,'pos'] == 0:
                #多头变空仓
                if ret.at[i -1,'pos'] == 1:
                    close_pos_num+=1
                    ret.at[i, 'out_price'] = ret.at[i,'open']*(1-slippage)
                    ret.at[i, 'hold_num'] = 0
                    ret.at[i, 'commission_cost'] = ret.at[i-1,'hold_num']*ret.at[i, 'out_price']*commission_rate
                    trade_close.at[close_pos_num, 'close_pos_price'] = ret.at[i, 'out_price']
                    trade_close.at[close_pos_num, 'close_date'] = ret.at[i, 'date']
                    ret.at[i, 'static_equity'] = ret.at[i-1, 'static_equity']+(trade_close.at[close_pos_num, 'close_pos_price'] - trade_open.at[open_pos_num,'open_pos_price'])*ret.at[i-1,'hold_num']- ret.at[i, 'commission_cost']
                    
                #空头变空仓
                if ret.at[i -1,'pos'] == -1:
                    close_pos_num+=1
                    ret.at[i, 'out_price'] = ret.at[i,'open']*(1+slippage)
                    ret.at[i, 'hold_num'] = 0
                    ret.at[i, 'commission_cost'] = ret.at[i-1,'hold_num']*ret.at[i, 'out_price']*commission_rate
                    trade_close.at[close_pos_num, 'close_pos_price'] = ret.at[i, 'out_price']
                    trade_close.at[close_pos_num, 'close_date'] = ret.at[i, 'date']
                    ret.at[i, 'static_equity'] = ret.at[i-1, 'static_equity']+(trade_open.at[open_pos_num,'open_pos_price']-trade_close.at[close_pos_num, 'close_pos_price'])*ret.at[i-1,'hold_num']- ret.at[i, 'commission_cost']
                ret.at[i, 'short_margin'] = 0
                ret.at[i, 'long_margin'] = 0    
                ret.at[i, 'dynamic_equity'] = ret.at[i, 'static_equity'] 
                ret.at[i, 'cash'] = ret.at[i, 'dynamic_equity']
            
            if ret.at[i, 'dynamic_equity'] / ret.at[i, 'static_equity'] < (1 - stop_loss):  # 设置止损
                for m, j in enumerate(list(ret.index)[i:-1]):
                    if ret.at[j + 1, 'pos'] == ret.at[i, 'pos']:
                        ret.at[j + 1, 'pos'] = 0
                    else:
                        break
            if i == ret.index[-1]:
                # 多头平仓
                if ret.at[i -1,'pos'] == 1:
                    close_pos_num+=1
                    ret.at[i, 'out_price'] = ret.at[i,'open']*(1-slippage)
                    ret.at[i, 'hold_num'] = 0
                    ret.at[i, 'commission_cost'] = ret.at[i-1,'hold_num']*ret.at[i, 'out_price']*commission_rate
                    trade_close.at[close_pos_num, 'close_pos_price'] = ret.at[i, 'out_price']
                    trade_close.at[close_pos_num, 'close_date'] = ret.at[i, 'date']
                    ret.at[i, 'static_equity'] = ret.at[i-1, 'static_equity']+(trade_close.at[close_pos_num, 'close_pos_price'] - trade_open.at[open_pos_num,'open_pos_price'])*ret.at[i-1,'hold_num']- ret.at[i, 'commission_cost']
                # 空头平仓
                if ret.at[i -1,'pos'] == -1:
                    close_pos_num+=1
                    ret.at[i, 'out_price'] = ret.at[i,'open']*(1+slippage)
                    ret.at[i, 'hold_num'] = 0
                    ret.at[i, 'commission_cost'] = ret.at[i-1,'hold_num']*ret.at[i, 'out_price']*commission_rate
                    trade_close.at[close_pos_num, 'close_pos_price'] = ret.at[i, 'out_price']
                    trade_close.at[close_pos_num, 'close_date'] = ret.at[i, 'date']
                    ret.at[i, 'static_equity'] = ret.at[i-1, 'static_equity']+(trade_open.at[open_pos_num,'open_pos_price']-trade_close.at[close_pos_num, 'close_pos_price'])*ret.at[i-1,'hold_num']- ret.at[i, 'commission_cost']
                    ret.at[i, 'short_margin'] = 0
                ret.at[i, 'short_margin'] = 0
                ret.at[i, 'long_margin'] = 0    
                ret.at[i, 'dynamic_equity'] = ret.at[i, 'static_equity'] 
                ret.at[i, 'cash'] = ret.at[i, 'dynamic_equity']  
     
    # ===策略评价部分
    evaluate = pd.DataFrame()  # 记录评价指标的计算结果
    for i in range(1,close_pos_num + 1):
        # 交易成本
        trade_close.at[i, 'cost'] = (trade_open.at[i, 'open_pos_price'] + trade_close.at[i, 'close_pos_price']) *\
                trade_open.at[i, 'hold_num'] * commission_rate
        # 净利润
        # 多头建仓时
        if trade_open.at[i, 'type'] == 1:
                trade_close.at[i, 'net_margin'] = (trade_close.at[i, 'close_pos_price'] - trade_open.at[i, 'open_pos_price'])\
                    * trade_open.at[i, 'hold_num'] - trade_close.at[i, 'cost']
        # 空头建仓时
        if trade_open.at[i, 'type'] == -1:
            trade_close.at[i, 'net_margin'] = (trade_open.at[i, 'open_pos_price'] - trade_close.at[i, 'close_pos_price'])\
                * trade_open.at[i, 'hold_num'] - trade_close.at[i, 'cost']
        # 收益率 平仓时计算收益
        trade_close.at[i, 'return_rate'] = trade_close.at[i, 'net_margin'] / (trade_open.at[i, 'open_pos_price']
            * trade_open.at[i, 'hold_num'])
    # 交易胜率
    
    # evaluate.at[0, 'trade_win_ratio'] = np.sum(trade_close['net_margin'] > 0) / len(trade_close['net_margin'])
    # 盈亏比
#    if - np.sum(trade_close['net_margin'][trade_close['net_margin'] < 0])!=0:
#         evaluate.at[0, 'win_loss_ratio'] = np.sum(trade_close['net_margin'][trade_close['net_margin'] > 0]) / \
#                                               (- np.sum(trade_close['net_margin'][trade_close['net_margin'] < 0]))
        # 日胜利
    ret['daily_return'] = ret['dynamic_equity']/ret['dynamic_equity'].shift(1) - 1
    ret.at[begin,'daily_return'] = 0
    evaluate.at[0, 'daily_win_ratio'] = np.sum(ret['daily_return']>0)/len(ret['daily_return'])       
    
    # 周胜率                               
    week_ret = ret[['date', 'dynamic_equity']].set_index('date').resample('W').last()
    week_ret['weekly_return'] = week_ret['dynamic_equity']/week_ret['dynamic_equity'].shift(1) - 1
    week_ret['weekly_return'][0]=0
    evaluate.at[0, 'week_win_ratio'] = np.sum(week_ret['weekly_return']>0)/len(week_ret['weekly_return'])
    
     # 月胜率
    month_ret = ret[['date', 'dynamic_equity']].set_index('date').resample('M').last()
    month_ret['month_return'] = month_ret['dynamic_equity'] / month_ret['dynamic_equity'].shift(1) - 1
    evaluate.at[0, 'month_win_ratio'] = np.sum(month_ret['month_return'] > 0) / len(month_ret['month_return']) 
    
    # 策略总收益率
    evaluate.at[0, 'total_return'] = ret['dynamic_equity'].iloc[-1]/ret['dynamic_equity'][0] - 1
    # 策略年化收益率
    evaluate.at[0, 'annual_return'] = (evaluate.at[0, 'total_return']+1)**(250/len(ret)) -1 
    
    # 基准年化收益率
    base_return = ret['close'].iloc[-1] / ret['close'].iloc[0] - 1
    evaluate.at[0, 'base_annual_return'] = (base_return + 1) ** (250 / ret.shape[0]) - 1
    
    # 回撤比例
    for i in list(ret.index):
        c = np.max(ret['dynamic_equity'][:i + 1])
        ret.at[i, 'back_ratio'] = (ret.at[i, 'dynamic_equity'] - c) / c
    # 最大回撤
    evaluate.at[0, 'max_withdraw'] = np.max(-ret['back_ratio'])
    
    # 策略年化波动率
    evaluate.at[0, 'annual_std'] = ret['daily_return'].std() * 250 ** 0.5
    
    # 夏普比率
    evaluate.at[0, 'sharp_ratio'] = (evaluate.at[0, 'annual_return'] - Rf) / evaluate.at[0, 'annual_std']
    return ret, evaluate

#def equity_curve_simple(Rf=0.035):
#    """
#    不考虑手续费和滑点的资金曲线计算方式
#    :param df: 要求输入的DataFrame已包含每日仓位
#    :param Rf: 无风险收益率，用来计算夏普比率
#    :return:
#    """
#    # ret = df[['date', 'pos', 'open', 'close']].copy()  # 用来存储计算出的参数的表格
#    # ret = pd.read_excel(r'F:\2018\实习\实习资料\machine_learning\ret_ada.xlsx')
#    ret = pd.read_excel(r'F:\2018\实习\实习资料\machine_learning\ret_gbdt.xlsx')
#    # ret = pd.read_excel(r'F:\2018\实习\实习资料\machine_learning\ret_xgboost.xlsx')
#    ret['pct_chg'] = ret['close'] / ret['close'].shift(1) - 1  # 计算标的资产每日涨跌幅
#    ret.at[ret.index[0], 'pct_chg'] = 0  # 将第一天涨跌幅设为0
#
#    # ===计算收益曲线
#    # 当天空仓时，pos为0，收益为0
#    # 当天多头时，pos为1，收益为资产本身的涨跌幅
#    # 当天空头时，pos为-1，收益为负的资产本身的涨跌幅
#    ret['daily_return'] = ret['pct_chg'] * ret['pos']  # 计算每日收益
#    ret['dynamic_equity'] = (ret['daily_return'] + 1).cumprod()  # 根据每日收益计算累计收益（复利投资）
#
#    # ===策略评价部分
#    evaluate = pd.DataFrame()  # 记录评价指标计算结果
#
#    # 日胜利
#    evaluate.at[0, 'daily_win_ratio'] = np.sum(ret['daily_return'] > 0) / len(ret['daily_return'])  # 盈利天数/总天数
#
#    # 周胜率
#    ret['date'] = pd.to_datetime(ret['date'])  # 将字符串表示的日期转换成时间戳
#    week_ret = ret[['date', 'dynamic_equity']].set_index('date').resample('W').last()  # 取每周最后一天净值
#    week_ret['week_return'] = week_ret['dynamic_equity'] / week_ret['dynamic_equity'].shift(1) - 1  # 计算每周收益
#    evaluate.at[0, 'week_win_ratio'] = np.sum(week_ret['week_return'] > 0) / len(week_ret['week_return'])  # 盈利周数/总周数
#
#    # 月胜率
#    month_ret = ret[['date', 'dynamic_equity']].set_index('date').resample('M').last()  # 取每月最后一天净值
#    month_ret['month_return'] = month_ret['dynamic_equity'] / month_ret['dynamic_equity'].shift(1) - 1  # 计算每月收益
#    evaluate.at[0, 'month_win_ratio'] = np.sum(month_ret['month_return'] > 0) / len(month_ret['month_return'])  # 盈利月数/总月数
#
#    # 策略总收益率
#    evaluate.at[0, 'total_return'] = ret['dynamic_equity'].iloc[-1] / ret['dynamic_equity'].iloc[0] - 1
#
#    # 策略年化收益率
#    evaluate.at[0, 'annual_return'] = (evaluate.at[0, 'total_return'] + 1) ** (250 / ret.shape[0]) - 1  # 假设每年250个交易日
#    
#    # 基准年化收益率
#    base_total_return = ret['close'].iloc[-1] / ret['close'].iloc[0] - 1
#    evaluate.at[0, 'base_annual_return'] = (base_total_return + 1) ** (250 / ret.shape[0]) - 1
#
#    # 回撤比例
#    for i in list(ret.index):
#        c = np.max(ret['dynamic_equity'][:i + 1])
#        ret.at[i, 'back_ratio'] = (ret.at[i, 'dynamic_equity'] - c) / c  # 当日净值相对于最高点的回撤比例
#
#    # 最大回撤
#    evaluate.at[0, 'max_withdraw'] = np.max(-ret['back_ratio'])
#
#    # 策略年化波动率
#    evaluate.at[0, 'annual_std'] = ret['daily_return'].std() * 250 ** 0.5  # 假设每年250个交易日
#
#    # 夏普比率
#    evaluate.at[0, 'sharp_ratio'] = (evaluate.at[0, 'annual_return'] - Rf) / evaluate.at[0, 'annual_std']
#
#    return ret, evaluate    


# adabost
evaluate_save = pd.DataFrame()
for i in sorted(commodity):
    ret_ada = pd.read_csv(open('F:/2018/实习/实习资料/machine_learning/ret_save/'+'xgboost_'+str(i)+'.csv'),index_col = 0,parse_dates = [1]) # 读入交易信号
    ret_ada1 = ret_ada.copy()
    ret1,evaluate1 = equity_curve(ret_ada1,initial_money=1000000, slippage=5.0/10000, commission_rate=5.0/10000,
                 trading_units=1, margin_ratio=0.1,stop_loss=1, Rf=0.035)
    evaluate1['commodity'] = i
    # print(evaluate1)
    evaluate_save = evaluate_save.append(evaluate1,ignore_index = True)


for i in ['AU','FG','JM','JR','NI','PM','RM','SN']:
    ret_ada = pd.read_csv(open('F:/2018/实习/实习资料/machine_learning/ret_save/'+'xgboost_'+i+'.csv'),index_col = 0,parse_dates = [1])    
    ret_ada1 = ret_ada.copy()
    ret1,evaluate1 = equity_curve(ret_ada1,initial_money=1000000, slippage=5.0/10000, commission_rate=5.0/10000,
                     trading_units=1, margin_ratio=0.1,stop_loss=1, Rf=0.035)
## 复利净值曲线和回撤比例图

    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    x = np.array(ret1['date'])
    y1 = np.array(ret1['dynamic_equity']/1000000)
    y2 = np.array(ret1['back_ratio'])
    fig= plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.stackplot(x, y2,-0.15-y2, colors = ['lightgrey','w'],labels = [u'回测比例',None])
    ax1.legend(loc= 8,bbox_to_anchor=(0.8,0))
    ax1.set_xlabel('日期')
    ax1.set_ylabel('回撤比例')
    ax2 = ax1.twinx() # this is the important function
    ax2.plot(x,y1,c = 'r',label= u'账户金额')
    ax2.legend(loc=8,bbox_to_anchor=(0.8,0.1))
    ax2.set_ylabel('账户金额')
    plt.gcf().autofmt_xdate()# 使日期变斜
    plt.title(i)
    plt.show()
    #plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    x = np.array(ret1['date'])
    y1 = ret1['close']/ret1['close'][0]
    y2 = ret1['dynamic_equity']/1000000
    #y3 = ret3['dynamic_equity']/1000000
    plt.plot(x,y1,label = '大宗商品收盘价净值')
    plt.plot(x,y2,label = '策略净值曲线')
    #plt.plot(x,y3,label = '策略改进后净值曲线')
    plt.legend()
    plt.xlabel('日期')
    plt.ylabel('净值')
    plt.gcf().autofmt_xdate()# 使日期变斜
    plt.title(i)
    plt.show()




