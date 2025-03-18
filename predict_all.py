import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow import keras
from tqdm import tqdm  # 用于显示进度条
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# 设置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# 创建全局归一化器
scaler = preprocessing.StandardScaler()

# 定义CSV文件路径
file_path = 'A股日度交易数据-米筐-2000-01-01-2024-11-22.csv'
output_path = 'output.csv'

# 自定义参数
# 定义时间窗大小
WINDOW_SIZE = 20  # 确保这个变量被定义

# 定义训练集比例
TRAIN_RATIO = 0.8  # 80%用于训练，20%用于测试

# 定义训练轮数
EPOCHS = 200 # 根据需求调整

# 定义dropout系数
DROPOUT_RATE = 0.1

# 定义批处理大小
BATCH_SIZE = 32

# 定义精度
ACCURACY = 0.1

# 禁用GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_all_data(file_path):
    """
    一次性加载所有需要的列的数据，并进行缺失值填补和归一化，返回按股票代码分组的数据字典。
    """
    required_columns = ['code', 'date', 'open', 'high', 'low', 'close',
                        'volume', 'amount', 'size', 'num_trades',
                        'prev_close', 'limit_up', 'limit_down', 'ret_o2c', 'ret_c2o']
    try:
        # 加载数据
        data = pd.read_csv(
            file_path,
            usecols=required_columns,
            parse_dates=['date'],
            encoding='utf-8'
        )

        # 填补缺失值
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'amount',
                           'size', 'num_trades', 'prev_close', 'limit_up',
                           'limit_down', 'ret_o2c', 'ret_c2o']
        
        # 归一化
        data[feature_columns] = scaler.fit_transform(data[feature_columns])

        # 按股票代码分组
        grouped_data = data.groupby('code')

        # 将分组数据转换为字典
        stock_data_dict = {}
        for stock_code, group in grouped_data:
            stock_data_dict[stock_code] = group.reset_index(drop=True)

        return stock_data_dict
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None

def create_sequences(data, window_size=5):
    """
    根据时间窗创建时序数据集。
    """
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size][-2])  # 假设预测 'ret_o2c'，根据需求调整
    return np.array(X), np.array(y)

def mape(y_true, y_pred):
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100

def build_model(window, feanum, dropout_rate):
    model = keras.Sequential([
        keras.layers.Input(shape=(window, feanum)),
        keras.layers.LSTM(256, return_sequences=True),  # 增加到256单元
        keras.layers.Dropout(dropout_rate),
        keras.layers.LSTM(128, return_sequences=True),  # 增加到128单元
        keras.layers.Dropout(dropout_rate),
        keras.layers.LSTM(64, return_sequences=False),  # 增加到64单元
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(32, activation='relu'),      # 增加到32神经元
        keras.layers.Dense(16, activation='relu'),      # 保留中间层
        keras.layers.Dense(1, activation='linear')
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['mae'])
    return model

def plot_prediction_vs_actual(y_true, y_pred, stock_code, dates=None):
    """
    绘制预测值与实际值的对比图
    """
    plt.figure(figsize=(12, 6))
    
    # 如果有日期数据，则使用日期作为x轴
    if dates is not None and len(dates) == len(y_true):
        x = dates
        plt.xlabel('日期')
        # 设置x轴日期格式
        plt.gcf().autofmt_xdate()
    else:
        x = range(len(y_true))
        plt.xlabel('测试样本')
    
    # 绘制真实值
    plt.plot(x, y_true, label='实际值', color='blue', linewidth=2)
    
    # 绘制预测值
    plt.plot(x, y_pred, label='预测值', color='red', linewidth=1, linestyle='--')
    
    # 添加标签和标题
    plt.title(f'股票 {stock_code} 预测结果对比')
    plt.ylabel('收益率 (ret_o2c)')
    plt.legend()
    plt.grid(True)
    
    # 计算相关性系数
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    plt.figtext(0.15, 0.85, f'相关系数: {correlation:.4f}', 
                bbox=dict(facecolor='white', alpha=0.5))
    
    # 保存图表
    plt.savefig(f'{stock_code}_prediction.png', dpi=300)
    print(f"已保存预测图表到 {stock_code}_prediction.png")
    plt.show()

def stock_prediction(stock_code, stock_data_dict, is_first_stock=False):
    """
    训练和预测单个股票的数据。
    如果 is_first_stock=True，则绘制预测结果图表
    """
    window, train_ratio, epochs, dropout_rate = (WINDOW_SIZE, TRAIN_RATIO, EPOCHS, DROPOUT_RATE)
    try:
        # 获取股票数据
        df_stock = stock_data_dict.get(stock_code)
        if df_stock is None or df_stock.empty:
            print(f"股票代码 {stock_code} 的数据为空。")
            return (stock_code, [None] * 7)
        
        # 准备特征和标签
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'amount',
                          'size', 'num_trades', 'prev_close', 'limit_up',
                          'limit_down', 'ret_o2c', 'ret_c2o']

        # 检查数据中是否存在NaN值
        if df_stock[feature_columns].isnull().any().any():
            print(f"股票代码 {stock_code} 的数据包含NaN值，将进行填充。")
            # 填充NaN值
            df_stock[feature_columns] = df_stock[feature_columns].fillna(0)

        df_features = df_stock[feature_columns].values
        
        # 存储原始日期，用于绘图
        dates = None
        if 'date' in df_stock.columns:
            dates = df_stock['date'].values

        # 创建序列
        X, y = create_sequences(df_features, window_size=window)
        if len(X) == 0:
            print(f"股票代码 {stock_code} 的数据长度不足以创建序列。")
            return (stock_code, [None] * 7)
        
        train_size = int(len(X) * train_ratio)

        # 划分训练集和测试集
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 如果有日期数据，也相应划分
        test_dates = None
        if dates is not None:
            # 由于创建序列时，数据点会减少window_size个，因此需要调整日期索引
            # 测试集对应的日期应该从 train_size + window 开始
            test_dates = dates[train_size + window:]
            if len(test_dates) > len(y_test):
                test_dates = test_dates[:len(y_test)]

        # 检查训练数据中是否存在NaN值
        if np.isnan(X_train).any() or np.isnan(y_train).any():
            print(f"股票代码 {stock_code} 的训练数据包含NaN值，将替换为0。")
            X_train = np.nan_to_num(X_train)
            y_train = np.nan_to_num(y_train)

        if np.isnan(X_test).any() or np.isnan(y_test).any():
            print(f"股票代码 {stock_code} 的测试数据包含NaN值，将替换为0。")
            X_test = np.nan_to_num(X_test)
            y_test = np.nan_to_num(y_test)

        # 定义特征数量
        feanum = X_train.shape[2]

        # 构建模型
        model = build_model(window, feanum, dropout_rate)

        # 添加早停法和NaN监控
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        
        # 添加学习率调度器
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )

        # 训练模型
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=[early_stop, lr_scheduler],  # 添加学习率调度器
            verbose=0  # 设置为0以减少输出
        )

        # 在训练集和测试集上的预测
        y_train_pred = model.predict(X_train, verbose=0)
        y_test_pred = model.predict(X_test, verbose=0)
        
        # 检查预测结果是否包含NaN值
        if np.isnan(y_train_pred).any():
            print(f"股票代码 {stock_code} 的训练预测结果包含NaN值。")
            # 替换NaN值为0
            y_train_pred = np.nan_to_num(y_train_pred)
            
        if np.isnan(y_test_pred).any():
            print(f"股票代码 {stock_code} 的测试预测结果包含NaN值。")
            # 替换NaN值为0
            y_test_pred = np.nan_to_num(y_test_pred)

        # 安全计算评估指标
        try:
            train_mae = mean_absolute_error(y_train, y_train_pred)
        except:
            print(f"计算训练集MAE出错，使用默认值。")
            train_mae = np.nan
        
        try:
            train_mse = mean_squared_error(y_train, y_train_pred)
        except:
            print(f"计算训练集MSE出错，使用默认值。")
            train_mse = np.nan
            
        try:
            train_mape = mape(y_train, y_train_pred)
        except:
            print(f"计算训练集MAPE出错，使用默认值。")
            train_mape = np.nan
        
        try:
            test_mae = mean_absolute_error(y_test, y_test_pred)
        except:
            print(f"计算测试集MAE出错，使用默认值。")
            test_mae = np.nan
            
        try:
            test_mse = mean_squared_error(y_test, y_test_pred)
        except:
            print(f"计算测试集MSE出错，使用默认值。")
            test_mse = np.nan
            
        try:
            test_mape = mape(y_test, y_test_pred)
        except:
            print(f"计算测试集MAPE出错，使用默认值。")
            test_mape = np.nan

        results = [train_mae, train_mse, train_mape, test_mae, test_mse, test_mape]
    
        if len(y_test) < 2:
            accuracy = None  # 数据不足以计算涨跌正确率
        else:
            # 误差在指定阈值内
            correct_count = []
            try:
                # 尝试反归一化，这里要小心处理
                y_test_2d = y_test.reshape(-1, 1)
                y_test_pred_2d = y_test_pred.reshape(-1, 1)
                
                # 创建临时数组，与原始特征数量相同
                temp_test = np.zeros((len(y_test), len(feature_columns)))
                temp_test[:, -2] = y_test  # 假设ret_o2c是倒数第二列
                
                temp_pred = np.zeros((len(y_test_pred), len(feature_columns)))
                temp_pred[:, -2] = y_test_pred.flatten()  # 假设ret_o2c是倒数第二列
                
                # 反归一化
                y_test_rare = scaler.inverse_transform(temp_test)[:, -2]
                y_test_pred_rare = scaler.inverse_transform(temp_pred)[:, -2]
                
                # 如果是第一只股票，绘制图表
                if is_first_stock:
                    try:
                        # 过滤掉任何无效值
                        mask = ~np.isnan(y_test_rare) & ~np.isnan(y_test_pred_rare) & \
                               ~np.isinf(y_test_rare) & ~np.isinf(y_test_pred_rare)
                        
                        if np.sum(mask) > 0:
                            valid_y_test = y_test_rare[mask]
                            valid_y_pred = y_test_pred_rare[mask]
                            valid_dates = test_dates[mask] if test_dates is not None else None
                            
                            # 绘制图表
                            plot_prediction_vs_actual(valid_y_test, valid_y_pred, stock_code, valid_dates)
                    except Exception as e:
                        print(f"绘制图表时出错: {e}")
                
                # 检查反归一化后的值是否包含 NaN 或 无穷值
                mask = ~np.isnan(y_test_rare) & ~np.isnan(y_test_pred_rare) & \
                       ~np.isinf(y_test_rare) & ~np.isinf(y_test_pred_rare) & \
                       (y_test_rare != 0)  # 避免除以零
                
                if np.sum(mask) > 0:
                    for i in range(len(y_test)):
                        if not mask[i]:
                            continue
                        if abs((y_test_pred_rare[i] - y_test_rare[i]) / y_test_rare[i]) < ACCURACY:
                            correct_count.append(1)
                        else:
                            correct_count.append(0)
                
                accuracy = sum(correct_count) / len(correct_count) * 100 if correct_count else None
            except Exception as e:
                print(f"计算准确率时出错: {e}")
                accuracy = None
                
        results.append(accuracy)
        return (stock_code, results)
    except Exception as e:
        print(f"处理股票代码 {stock_code} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return (stock_code, [None] * 7)

def main():
    # 加载并预处理数据
    print("正在加载并预处理数据...")
    stock_data_dict = load_all_data(file_path)
    if stock_data_dict is None:
        print("无法加载数据，程序退出。")
        return

    # 获取所有股票代码
    stock_codes = list(stock_data_dict.keys())
    print(f"共找到 {len(stock_codes)} 个股票代码。")

    # 测试前100只股票
    print("开始前10只股票的预测...")
    results_first_ten = []
    
    # 特别处理第一只股票，为其生成预测图表
    if stock_codes:
        print(f"处理第一只股票 {stock_codes[0]} 并生成预测图表...")
        first_result = stock_prediction(stock_codes[0], stock_data_dict, is_first_stock=True)
        results_first_ten.append(first_result)
    
    # 处理剩余的股票
    for stock_code in tqdm(stock_codes[1:10], desc="预测进度"):
        result = stock_prediction(stock_code, stock_data_dict)
        results_first_ten.append(result)
    
    print("正在保存前10只股票预测结果到 output_part10.csv...")
    with open('output_part10.csv', 'w', encoding='utf-8') as f:
        f.write('stock_code,train_mae,train_mse,train_mape,test_mae,test_mse,test_mape,accuracy\n')
        for code, result in results_first_ten:
            if all(v is not None for v in result):
                f.write(f"{code},{','.join(map(str, result))}\n")
                print(f"{code},{','.join(map(str, result))}\n")
            else:
                f.write(f"{code},,,,,,,\n")  # 对于出错的股票，留下空白
    
    ok = input("是否继续预测剩余股票？(y/n): ")
    if ok.lower() != "y":
        return
    
    # 预测全部股票
    print("继续预测全部股票...")
    results_all = []
    
    for stock_code in tqdm(stock_codes, desc="预测进度"):
        result = stock_prediction(stock_code, stock_data_dict)
        results_all.append(result)
    
    # 保存结果到CSV
    print("正在保存预测结果到 output.csv...")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('stock_code,train_mae,train_mse,train_mape,test_mae,test_mse,test_mape,accuracy\n')
        for code, result in results_all:
            if all(v is not None for v in result):
                f.write(f"{code},{','.join(map(str, result))}\n")
            else:
                f.write(f"{code},,,,,,,\n")  # 对于出错的股票，留下空白
    
    print(f"预测结果已保存至 {output_path} 文件。")

if __name__ == "__main__":
    main()