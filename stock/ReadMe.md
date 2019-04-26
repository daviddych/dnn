1. storage.py:  从网上下载指定的股票数据,保存到MySQL数据库
                (1). 首先确保本地安装MySQL数据库, 然后创建一个stock数据库
                (2). 运行该脚本从网络上抓取股票数据存入stock数据库中, stock_name存储股票名和代码, 股票数据存储在以股票名命名的数据表中


2. lstm_stock.py:  从MySQL数据库中读取指定的股票时序数据, 训练LSTM网络模型