import os
from cctools import CCTools


import tempfile

import cv2

CATS = {1:"Paragraph",2:"Head",3:"Footnote",4:"HeaderFooter",5:"Caption",6:"Table",7:"Figure",8:"Formula",9:"Wireless",10:"Form"}
all_docs = [
    # 一：含交互式元素的文档
    "One Loss for Quantization Deep Hashing",
    "Continual Object Detection via Prototypi",
    "A Comprehensive Catalog of UVIT Observations",
    "DSC20160400000_74147885",
    "Gov_Meeting",
    "gser2024134_152420702",
    "On axiomatizations of public announcemen",
    "quantum-notes",
    "基于ARIMA和BP神经网络的供应链需求预测",
    "基于CNN-BiLSTM-ARIMA模型的煤炭价格预测",

    # 二：含图像的文档
    "AN4539应用笔记",
    "Emergence of high-mass stars in complex fi",
    "LINFUSION：1 GPU, 1 MINUTE, 16K IMAGE",

    # 三：含复杂布局和设计的文档
    "高中物理力学笔记",

    # 四：含目录的文档
    "会计学原理财务会计_第一卷_14994",
    "全球数字经济白皮书（2022年）",
    "支持小微企业和个体工商户发展优惠政策指引",
    "深度学习入门",

    # 五：含表格和图表的文档
    "2455-Implicit Parameter-free Online Learnin",
    "The Formation of Milky Way \"Bones\"",

    # 六：混合内容文档
    "_积神_网络的_解释性研究综述",
    "1Dual Cross-Attention Learning for Fine-Gra",
    "02_knn_notes",
    "22-SimAN Exploring Self-Supervised Repres",
    "30-Recurrent Dynamic Embedding for Video",
    "824-LearningUnseenEmotionsfromGestures",
    "825-OptimizedPotentialInitializationforLow-",
    "7432-Sharp-MAML- Sharpness-Aware Mod",
    "gser2024134_82420715",
    "gser2024134_182420709",
    "临床预测模型基础",
    "大规模时间序列实现",

    # 七：纯文本文档（无）

    # 八：国标类-新
    "GB1903.20-2016食品安全国家标准食品营养强化剂硝酸硫胺素",
    "GB5009.34-2022食品安全国家标准食品中二氧化硫的测定",
    "GB5009.208-2016食品安全国家标准食品中生物胺的测定",
    "GB5009.296-2023食品安全国家标准食品中维生素D的测定",
    "GB31604.46-2023",
    "GB31604.59-2023",
    "GB+24406-2024",
    "GB+35848-2024",
    "GB+383615-2024",

    # 九：教育类-新
    "第1章",
    "第2章",
    "第3章",
    "第5章",
    "第6章",
    "第7章",
    "第8章",
    "第10章",
    "第12章",
    "计算机网络原理2018年版部分习题参考答案"
]


def api_call(imgpath,bbox):
    # id_map = {1:"Paragraph",2:"Head",3:"Footnote",4:"HeaderFooter",5:"Caption",6:"Table",7:"Figure",8:"Formula",9:"Wireless",10:"Form"}
    import requests

    table_url = 'https://demo-parser-tsr.connectedpdf.com/pipeline/predict'
    form_url = 'https://dev-parser-formclf.connectedpdf.com/classify/upload'
    table_map = {"wired":6,"wireless":9}
    form_map = {"wired":6,"form":10}
    # 查询参数

    table_data = {
        'boxes': str([bbox]),
    }
    imgpath = str(imgpath)
    # 文件和表单数据
    table_files = {
        'file': (os.path.basename(imgpath), open(imgpath, 'rb'), 'image/jpeg')
    }
    table_params = {'trace_id': 'ea01f9c4-5a54-460c-b257-6569774927df'}

    table_response = requests.post(table_url, params=table_params,files=table_files, data=table_data)

    if table_response.status_code == 200:
        if table_response.json()['return_code'] == 101:
            # 将路径写入到文件中
            with open("tmp.txt","w") as f:
                f.write(imgpath)
            return 6
        
        result = table_response.json()['content'][0]['type']
        
        if table_map[result] == 6:
            image = cv2.imread(imgpath)
            ann_img = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_path = temp_file.name
                cv2.imwrite(temp_path, ann_img)
            
                # 准备文件
                form_files = {
                    'file': (temp_path, open(temp_path, 'rb'), 'image/jpeg')
                }
                # 发送 POST 请求
                form_response = requests.post(form_url, headers={'accept': 'application/json'}, files=form_files)
                if form_response.status_code == 200:
                    result = form_response.json()['label']
                    if form_map[result] == 10:
                        return 10
                    else:
                        return 6
                else:
                    print("Request failed with status code:", table_response.status_code)
        else:
            # print(table_map[result])
            return 9
    else:
        print("Request failed with status code:", table_response.status_code)

def one(root):
    visual = False
    new_root = root.replace("raw","NEW")
    old_root = root.replace("raw","OLD")
    correct_root = root.replace("raw","CORRECT")
    os.makedirs(new_root,exist_ok=True)
    os.makedirs(old_root,exist_ok=True)
    os.makedirs(correct_root,exist_ok=True)
    
    # ccdata_file = CCTools(ROOT=root)
    # # 对齐类别
    # ccdata_file.update_cat(newCat=CATS)
    # if visual:
    #     ccdata_file.visual()
    
    # # 过滤公式以及table标签，保存到NEW目录
    # newData = ccdata_file.filter(imgs=all_docs,cats=["Table"],newObj=CCTools(ROOT=new_root),mod="or",visual=visual,level="img",sep_data=True)
    
    # # 保存分离后的旧数据，保存到OLD目录
    # ccdata_file.save(New=CCTools(ROOT=old_root),visual=visual)
    
    newData = CCTools(ROOT=new_root)
    newData.correct(api_url=api_call,cats=["Table"],newObj=CCTools(ROOT=correct_root),visual=visual)
    


def filter_to_correct(root="/home/zyj/github/CCTools/竞品/raw"):
    dirlist = os.listdir(root)
    for data_dir in dirlist:
        # if data_dir in ["含交互式元素的文档",  "含图像的文档",  "含复杂布局和设计的文档",  "含表格和图表的文档",  "国标文件-新",  "教育领域-新"]:
        #     continue
        if data_dir in ["含目录的文档"]:
            if os.path.isdir(os.path.join(root,data_dir)):
                one(os.path.join(root,data_dir))
                
def merge(root):
    dirlist = os.listdir(root)
    ccdata_files = []
    for data_dir in dirlist:
        data_path = os.path.join(root,data_dir)
        if not os.path.isdir(data_path):
            continue
        ccdata_files.append(CCTools(ROOT=data_path))
    
    newObj = ccdata_files[0]
    newObj.merge(newObj=ccdata_files[1:],cat_keep=True)
    newObj.save(New=CCTools(ROOT=root.replace("raw","MERGE")),visual=False)
    
            
    pass

def split(root):
    pass
    
    

if __name__ == "__main__":
    merge(root="/home/zyj/github/CCTools/竞品/raw")
