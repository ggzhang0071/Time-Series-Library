import pandas as pd
import os
from tqdm import tqdm
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

def query_data(query):

    username = 'user'
    password = 'user123'
    host='192.168.20.10'
    port = '3306'  # 默认 MySQL 端口

    database = '华电大模型'

    # 创建数据库连接字符串
    connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{database}'
    
    # 创建数据库引擎
    engine = create_engine(connection_string)
        
    # 使用 pd.read_sql 读取数据
    databases_df = pd.read_sql(query, engine)
    return databases_df

orders = {
    '输入00000002-大秦线检修时间': '结束时间',
    '输入00000003-电力集团招标': '招标日期',
    '输入00000004-易煤北方港指数_处理后': '发布日期',
    '输入00000005-易煤长江口指数_处理后': '发布日期',
    '输入00000006-CTCI曹妃甸指数_处理后': '发布日期',
    '输入00000008-珠投竞价': '发布日期',
    '输入00000012-找煤网动力煤参考价_处理后': '发布日期',
    '输入00000013-呼局批车数量统计_处理后': '批车日期',
    '输入00000014-国能坑口外购价（巴图塔）_处理后': '开始日期',
    '输入00000015-国家能源黄骅港下水煤销售价_处理后': '开始日期',
    '输入00000019-CCI指数_处理后': '发布日期',
    '输入00000048-国能580_物流园区_发运_库存数据_处理后': '发布日期',
    '输入00000049-国能580_运输价格_汽运_短途_处理后': '发布日期',
    '输入00000105-找煤AI数据_港口动态_北方港调运': '发布日期',
    '输入00000106-找煤AI数据_港口动态_国内港口库存': '发布日期',
    '输入00000107-找煤AI数据_终端动态_重点区域电厂': '发布日期',
    '输入00000108-找煤AI数据_终端动态_沿海六大电厂': '发布日期',
    '输入00000109-找煤AI数据_终端动态_其它电厂': '发布日期',
    '输入00000250-秦皇岛煤炭网-指数中心_OCFI煤炭海运费走势图_秦皇岛-张家港(2-3WD)': '报价日期',
    '输入00000251-秦皇岛煤炭网-指数中心_OCFI煤炭海运费走势图_秦皇岛-广州（6-7WD）': '报价日期',
    '输入00000252-API-年节假日': '日期',
    '输入00000268-CCTD-煤炭价格_鄂尔多斯市_伊金霍洛旗': '时间',
    '输入00000269-CCTD-煤炭价格_榆林市_神木县': '时间',
    '输入00000270-CCTD-煤炭价格_朔州_朔州市': '时间',
    '输入00000271-CCTD-煤炭价格_朔州_右玉县': '时间',
    '输入00000272-CCTD-煤炭价格_CCTD标志煤种价格_环渤海动力煤现货参考价': '时间',
    '输入00000273-CCTD-煤炭价格_到岸价格_印尼': '时间',
    '输入00000274-CCTD-煤炭价格_到岸价格_澳大利亚': '时间',
    '输入00000275-CCTD-煤炭价格_到岸价格_俄罗斯': '时间',
    '输入00000278-CCTD-煤炭运费_水运价格_进口煤炭运费':'日期',
    '输入00000279-CCTD-煤炭运费_水运价格_沿海煤炭运费':'时间',
    '输入00000280-人民币汇率中间价': '日期',
    '输入00000286-长江三峡通航管理局-三峡入库水位': '时间',
    '输入00000287-长江三峡通航管理局-三峡出库水位': '时间',
    '输入00000288-SGX-新加坡动力煤期货': '日期',
    '输入00000290-钢联-LNG：市场主流价：内蒙古（日）': '时间',
    '输入00000291-钢联-LNG：市场主流价：山西（日）':'时间',
    '输入00000292-钢联-LNG：市场主流价：陕西（日）': '时间',
    '输入00000294-钢联-尿素：产能利用率：中国（日）':'时间',
    '输入00000296-钢联-尿素：小颗粒：市场主流价：山东（日）':'时间',
    '输入00000297-钢联-尿素：小颗粒：市场主流价：山西（日）':'时间',
    '输入00000298-钢联-尿素：小颗粒：市场主流价：河北（日）':'时间',
    '输入00000299-钢联-尿素：小颗粒：市场主流价：河南（日）': '时间',
    '输入00000300-钢联-尿素：小颗粒：市场主流价：黑龙江（日）':'时间',
    '输入00000301-钢联-布伦特原油：FOB价：欧洲（日）': '时间',
    '输入00000302-钢联-水泥熟料：产能利用率：中国（周）': '时间',
    '输入00000303-钢联-水泥：价格指数（日）': '时间',
    '输入00000306-钢联-甲醇：国标：市场主流价：内蒙古（日）': '时间',
    '输入00000307-钢联-甲醇：国标：市场主流价：山西（日）': '时间',
    '输入00000308-钢联-甲醇：国标：市场主流价：陕西（日）': '时间',
    '输入00000309-钢联-辛塔、迪拜、布伦特原油：现货均价（日）':'时间'
    }

tbls = {
    '输入00000002-大秦线检修时间': ['开始时间','结束时间'],
    '输入00000003-电力集团招标': ['采购数量','招标日期','交货截止日'],
    '输入00000004-易煤北方港指数_处理后': ['发布日期','Q5000S0.8','Q5500S0.8'],
    '输入00000005-易煤长江口指数_处理后': ['发布日期','Q5000S0.8','Q5500S0.8'],
    '输入00000006-CTCI曹妃甸指数_处理后': ['发布日期','4500大卡','5000大卡','5500大卡'],
    '输入00000008-珠投竞价': ['发布日期','煤种','起拍价','销售量','硫分','热值','交货开始时间','交货结束时间'],
    '输入00000012-找煤网动力煤参考价_处理后': ['发布日期','4500K 1.0S','5000K 0.8S','5500K 0.8S'],
    '输入00000013-呼局批车数量统计_处理后': ['批车日期','列数'],
    '输入00000014-国能坑口外购价（巴图塔）_处理后': '*',
    '输入00000015-国家能源黄骅港下水煤销售价_处理后': '*',
    '输入00000019-CCI指数_处理后': '*',
    '输入00000048-国能580_物流园区_发运_库存数据_处理后': '*',
    '输入00000049-国能580_运输价格_汽运_短途_处理后': '*',
    '输入00000105-找煤AI数据_港口动态_北方港调运': ['发布日期','港口','卸车','调进','装船','下锚船','预报船舶'],
    '输入00000106-找煤AI数据_港口动态_国内港口库存': ['发布日期','区域','港口','动力煤','焦煤'],
    '输入00000107-找煤AI数据_终端动态_重点区域电厂': ['发布日期','省份','类型','数量'],
    '输入00000108-找煤AI数据_终端动态_沿海六大电厂': ['发布日期','电厂名称','库存量','日耗','可用天数'],
    '输入00000109-找煤AI数据_终端动态_其它电厂': ['发布日期','电厂名称','库存量','日耗','可用天数'],
    '输入00000250-秦皇岛煤炭网-指数中心_OCFI煤炭海运费走势图_秦皇岛-张家港(2-3WD)': ['报价日期','指数/航线','船型','单位','本期','期环比','年度同比'],
    '输入00000251-秦皇岛煤炭网-指数中心_OCFI煤炭海运费走势图_秦皇岛-广州（6-7WD）': ['报价日期','指数/航线','船型','单位','本期','期环比','年度同比'],
    '输入00000252-API-年节假日': ['日期','是否节假日'],
    '输入00000268-CCTD-煤炭价格_鄂尔多斯市_伊金霍洛旗': ['时间','报价地区','产品名称','灰份','挥发份','硫份','粘结指数','发热量','价格(元/吨)'],
    '输入00000269-CCTD-煤炭价格_榆林市_神木县': ['时间','报价地区','产品名称','灰份','挥发份','硫份','粘结指数','发热量','价格(元/吨)'],
    '输入00000270-CCTD-煤炭价格_朔州_朔州市': ['时间','报价地区','产品名称','灰份','挥发份','硫份','粘结指数','发热量','价格(元/吨)'],
    '输入00000271-CCTD-煤炭价格_朔州_右玉县': ['时间','报价地区','产品名称','灰份','挥发份','硫份','粘结指数','发热量','价格(元/吨)'],
    '输入00000272-CCTD-煤炭价格_CCTD标志煤种价格_环渤海动力煤现货参考价': ['时间','Q5500K','Q5000K','Q4500K'],
    '输入00000273-CCTD-煤炭价格_到岸价格_印尼': ['时间','产品名称','价格(元/吨)','发热量','到岸港口'],
    '输入00000274-CCTD-煤炭价格_到岸价格_澳大利亚': ['时间','产品名称','价格(元/吨)','发热量','到岸港口'],
    '输入00000275-CCTD-煤炭价格_到岸价格_俄罗斯': ['时间','产品名称','价格(元/吨)','发热量','到岸港口'],
    '输入00000278-CCTD-煤炭运费_水运价格_进口煤炭运费':['日期','数据类目','当期值(美元/吨)','环比变化'],
    '输入00000279-CCTD-煤炭运费_水运价格_沿海煤炭运费':['时间', '航线名称', '本期(元/吨)', '上期(元/吨)', '环比增减(元/吨)', '100 * (`本期(元/吨)` / `上期(元/吨)` - 1) as 环比增幅' ],
    '输入00000280-人民币汇率中间价': ['日期', '美元', '欧元', '日元', '港元', '英镑', '澳元', '新西兰元', '新加坡元', '瑞士法郎', '加元','林吉特', '卢布', '兰特', '韩元', '迪拉姆', '里亚尔', '福林', '兹罗提', '丹麦克朗', '瑞典克朗','挪威克朗', '里拉', '比索', '泰铢', '澳门元'],
    '输入00000286-长江三峡通航管理局-三峡入库水位': ['时间','水位','涨落'],
    '输入00000287-长江三峡通航管理局-三峡出库水位': ['时间','水位','涨落'],
    '输入00000288-SGX-新加坡动力煤期货': ['合约','结算价','日期'],
    '输入00000290-钢联-LNG：市场主流价：内蒙古（日）': ['时间','LNG：市场主流价：内蒙古（日）'],
    '输入00000291-钢联-LNG：市场主流价：山西（日）':['时间','LNG：市场主流价：山西（日）'],
    '输入00000292-钢联-LNG：市场主流价：陕西（日）': ['时间','LNG：市场主流价：陕西（日）'],
    '输入00000294-钢联-尿素：产能利用率：中国（日）':['时间','尿素：产能利用率：中国（日）'],
    '输入00000296-钢联-尿素：小颗粒：市场主流价：山东（日）':['时间','尿素：小颗粒：市场主流价：山东（日）'],
    '输入00000297-钢联-尿素：小颗粒：市场主流价：山西（日）':['时间','尿素：小颗粒：市场主流价：山西（日）'],
    '输入00000298-钢联-尿素：小颗粒：市场主流价：河北（日）':['时间','尿素：小颗粒：市场主流价：河北（日）'],
    '输入00000299-钢联-尿素：小颗粒：市场主流价：河南（日）': ['时间','尿素：小颗粒：市场主流价：河南（日）'],
    '输入00000300-钢联-尿素：小颗粒：市场主流价：黑龙江（日）':['时间','尿素：小颗粒：市场主流价：黑龙江（日）'],
    '输入00000301-钢联-布伦特原油：FOB价：欧洲（日）': ['时间','布伦特原油：FOB价：欧洲（日）'],
    '输入00000302-钢联-水泥熟料：产能利用率：中国（周）': ['时间','水泥熟料：产能利用率：中国（周）'],
    '输入00000303-钢联-水泥：价格指数（日）': ['时间','水泥：价格指数（日）'],
    '输入00000306-钢联-甲醇：国标：市场主流价：内蒙古（日）': ['时间','甲醇：国标：市场主流价：内蒙古（日）'],
    '输入00000307-钢联-甲醇：国标：市场主流价：山西（日）': ['时间','甲醇：国标：市场主流价：山西（日）'],
    '输入00000308-钢联-甲醇：国标：市场主流价：陕西（日）': ['时间','甲醇：国标：市场主流价：陕西（日）'],
    '输入00000309-钢联-辛塔、迪拜、布伦特原油：现货均价（日）':['时间','辛塔、迪拜、布伦特原油：现货均价（日）'],
    }

for x in tqdm(tbls.keys()):
    cols = tbls[x]
    combine_cols = [x for x in cols if 'as' not in x]
    add_cols = [x for x in cols if 'as' in x]
    if type(cols) is list:
        select_cols = ','.join([f'`{x}`' for x in combine_cols] +  [f'{x}' for x in add_cols])
    else:
        select_cols = cols
    order_k  = orders[x]
    select_str = f'select {select_cols} from `{x}` order by {order_k} desc;'
    data = query_data(select_str)
    print (x)
    #display(data.head())
    data.to_csv(f'orginal_file/{x}.csv',index=False)

query_data("select * from `输入00000250-秦皇岛煤炭网-指数中心_OCFI煤炭海运费走势图_秦皇岛-张家港(2-3WD)` where `报价日期` = '2020-01-02'")
