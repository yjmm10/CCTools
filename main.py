import os
from pathlib import Path
from cctools import CCTools


import tempfile

import cv2

from cctools.pipeline import Pipeline

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
    try:
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
    except:
        return None

def one(root):
    visual = False
    new_root = root.replace("raw","NEW")
    old_root = root.replace("raw","OLD")
    correct_root = root.replace("raw","CORRECT")
    os.makedirs(new_root,exist_ok=True)
    os.makedirs(old_root,exist_ok=True)
    os.makedirs(correct_root,exist_ok=True)
    
    ccdata_file = CCTools(ROOT=root)
    # 对齐类别
    ccdata_file.update_cat(newCat=CATS)
    if visual:
        ccdata_file.visual()
    
    # result = {}
    # result["原始的数据"] = ccdata_file.static()['total_images']
    # result["原始的图片"] = len(os.listdir(os.path.join(root,"images")))
    
    # 过滤公式以及table标签，保存到NEW目录
    newData = ccdata_file.filter(imgs=all_docs,cats=["Table"],newObj=CCTools(ROOT=new_root),mod="or",visual=visual,level="img",sep_data=True)
    # 保存分离后的旧数据，保存到OLD目录
    oldData = CCTools(ROOT=old_root)
    ccdata_file.save(New=oldData,visual=visual)
    # result["保留的数据"] = oldData.static()['total_images']
    # result["保留的图片"] = len(os.listdir(os.path.join(oldData.ROOT,"images")))
    # result["过滤的数据"] = newData.static()['total_images']
    # result["过滤的图片"] = len(os.listdir(os.path.join(newData.ROOT,"images")))
    
    
    
    correctData = newData.correct(api_url=api_call,cats=["Table"],newObj=CCTools(ROOT=correct_root),visual=visual)
    # result["纠正的数据"] = correctData.static()['total_images']
    # result["纠正的图片"] = len(os.listdir(os.path.join(correctData.ROOT,"images")))
    # print(root, result)
    pass

    


def filter_to_correct(root="竞品/raw"):
    dirlist = os.listdir(root)
    for data_dir in dirlist:
        if os.path.isdir(os.path.join(root,data_dir)):
            one(os.path.join(root,data_dir))
                
def merge():
    root="竞品/test"
    dirlist = os.listdir(root)
    ccdata_files = []
    for data_dir in dirlist:
        data_path = os.path.join(root,data_dir)
        if not os.path.isdir(data_path):
            continue
        ccdata_files.append(CCTools(ROOT=data_path))
    
    
    newObj = CCTools(ROOT=root.replace("test","MERGE"))
    newObj.merge(others=ccdata_files,cat_keep=True,newObj=newObj)
    newObj.save(New=newObj,visual=True)
    
    # srcObj = CCTools(ROOT=root.replace("test","test_MERGE"))
    # newObj.save(New=srcObj,visual=True)
    
    # srcObj = CCTools(ROOT="/home/zyj/github/CCTools/竞品/CORRECT/含图像的文档")
    # splitObj = CCTools(ROOT="/home/zyj/github/CCTools/竞品/SPILT/含图像的文档")
    # srcObj.split(ratio=[0.7,0.2,0.1],by_file=True,newObj=splitObj)


    pass

def split(root):
    pass
    
def static():
    root = "竞品"
    res = []
    for  dd  in ["raw","CORRECT","OLD"]:
        temp_root = os.path.join(root,dd)
        dirlist = os.listdir(temp_root)
        ccdata_files = {}
        for data_dir in dirlist:
            data_dir = os.path.join(temp_root,data_dir)
            if not os.path.isdir(data_dir):
                continue
            try:
                ccdata = CCTools(ROOT=data_dir)
                ann_imgs = ccdata.static()['total_images']
                ann_anns = ccdata.static()['total_annotations']
                dir_imgs = os.listdir(os.path.join(ccdata.ROOT,"images"))
                ccdata_files[data_dir] = {'标注图像':ann_imgs,'标注bbox':ann_anns,'目录图像':len(dir_imgs)}
            except Exception as e:
                continue
            
        res.append(ccdata_files)
    print(res)
    


def pipeline_custom():
    pipeline = Pipeline()
    data_root = Path("idp_q4")
    result,stats = pipeline.static_data(data_root,target_dirs=["annotations","images"],data_match={"instances_default_1.json":"images","instances_default.json":"images","instances_default.json":"images"},static_file=data_root.joinpath("static.xlsx"))


def diff_image(img1_path, img2_path, output_path):
    """
    计算两张图片的像素差值并保存
    
    Args:
        img1_path: 第一张图片路径
        img2_path: 第二张图片路径  
        output_path: 输出差值图路径
    """
    import cv2
    import numpy as np
    
    # 读取两张图片
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # 确保两张图片尺寸一致
    # assert img1.shape == img2.shape, print(img1.shape, img2.shape)
    if img1.shape != img2.shape:
        if img1.shape[0] * img1.shape[1] < img2.shape[0] * img2.shape[1]:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        else:
            img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    
    # 计算差值
    diff = cv2.absdiff(img1, img2)
    
    # 保存差值图
    cv2.imwrite(output_path, diff)


def IDP_dla_data(ROOT="idp_q4/Q4-Part2",ANNFILE="instances_default1.json",RATIO=[0.5,0.2,0.3],DST_ROOT="idp_q4/datasets",by_file=True,index=False,YOLOROOT="idp_q4/DLA_YOLO"):
    import shutil
    ROOT= Path(ROOT)
    DST_ROOT = Path(DST_ROOT)
    YOLOROOT = Path(YOLOROOT)
    IMGDIR = "images"
    ANNDIR = "annotations"
    
    DST_ROOT
    
    ccdata_file = CCTools(ROOT=ROOT, ANNFILE=ANNFILE, IMGDIR=IMGDIR, ANNDIR=ANNDIR)
    # 将公式标签替换为段落
    ccdata_file.rename_cat_in_ann(old_name='Formula',new_name='Paragraph')
    DST_ROOT = Path(DST_ROOT).joinpath(ROOT.stem+"_1") if index else Path(DST_ROOT).joinpath(ROOT.stem)
    YOLOPATH = YOLOROOT.joinpath(DST_ROOT.stem)
    
    shutil.rmtree(DST_ROOT,ignore_errors=True)
    empty_data = CCTools(ROOT=DST_ROOT, ANNFILE=ANNFILE, IMGDIR=IMGDIR, ANNDIR=ANNDIR,empty_init=True)
    pipeline = Pipeline()
    config = { 'cctools':ccdata_file,
                'ratio':RATIO,
                'by_file':by_file,
                'visual':False,
                'newObj':empty_data,
                'yolopath':YOLOPATH,
                }

    pipeline.split_data(config=config)
    
    ccdata_file._YOLO()
    YOLO_id2cls = dict(sorted(ccdata_file.YOLO_id2cls.items()))
    return YOLO_id2cls

# 统计.cache文件比例
def count_cache_files(root_dir):
    import json
    total_files = 0
    result = []
    for path in Path(root_dir).rglob('*'):
        if path.is_file():
            total_files += 1
            if path.suffix == '.cache':
                with open(path, "r") as f:
                    cache_data = json.load(f)
                    l_train = len(cache_data["Train"])
                    l_val = len(cache_data["Val"])
                    l_test = len(cache_data["Test"])
                    sum_all = l_train + l_val + l_test
                    result.append(f"{path}: {l_train}:{l_val}:{l_test}===>>> {l_train/sum_all:.2f}:{l_val/sum_all:.2f}:{l_test/sum_all:.2f}")
    for i in result:
        print(i)
        
def make_yolo_cfg(root_dir,id2cls,output_file="yolo_list.yaml"):
    """统计yolo目录下的list文件分布"""
    from collections import defaultdict
    result = defaultdict(list)
    
    for folder in Path(root_dir).iterdir():
        if folder.is_dir():
            for mod in ["Train","Test","Val"]:
                if mod in folder.name:
                    result[mod].append(folder.name)
    
    # 定义映射关系
    output_map = {
        "train": "Train",
        "val": "Val",
        "test": "Test"
    }
    with open(os.path.join(root_dir,output_file),"w") as f:
        f.write(f"path: {os.path.abspath(root_dir)}\n")
        for key, mod in output_map.items():
            result_list = result[mod] if mod else []  # 如果mod为空字符串，使用空列表
            f.write(f"{key}: {result_list}\n")
    
        f.write("names:\n")
        for id,cls in id2cls.items():
            f.write(f"  {id}: {cls}\n")
    
    print(f"yolo list file has been saved to {output_file}")
    
def idp_dla_pipeline(root_dir="idp_q4"):
    CVAT_ROOT = os.path.join(root_dir,"DLA_CVAT")
    DST_ROOT = os.path.join(root_dir,"DLA_COCO")
    YOLO_ROOT = os.path.join(root_dir,"DLA_YOLO")
    
    os.makedirs(DST_ROOT,exist_ok=True)
    os.makedirs(YOLO_ROOT,exist_ok=True)
    
    # 提取共同参数
    common_params = {
        "DST_ROOT": DST_ROOT,
        "YOLOROOT": YOLO_ROOT
    }

    # 默认配置参数
    default_params = {
        "ANNFILE": "instances_default.json",
        "RATIO": [0.7, 0.3]
    }

    # 特殊配置参数
    special_configs = {
        "Q4-Part2": {
            "ANNFILE": ["instances_default.json","instances_default_1.json"],
            "RATIO": [[0.5, 0.2, 0.3],[0.5, 0.2, 0.3]],
            "index": [False,True]
        },

        "Q4-Part3_Align": {
            "by_file": False
        }
    }

    # 遍历目录下所有文件夹
    for folder in Path(CVAT_ROOT).iterdir():
        if folder.is_dir() and not folder.name.startswith('.'):
            # 跳过输出目录
            if folder.name in ["DLA_COCO", "DLA_YOLO"]:
                continue
                
            # 基础配置
            config = {
                "ROOT": str(folder)
            }
            
            # 合并默认参数
            config.update(default_params)
            
            # 如果有特殊配置则更新
            if folder.name in special_configs:
                special_config = special_configs[folder.name]
                
                # 检查是否有多个配置
                if any(isinstance(v, list) for v in special_config.values()):
                    # 获取列表长度
                    list_lens = [len(v) for v in special_config.values() if isinstance(v, list)]
                    num_configs = list_lens[0] if list_lens else 1
                    
                    # 对每个配置执行处理
                    for i in range(num_configs):
                        curr_config = {}
                        for k,v in special_config.items():
                            if isinstance(v, list):
                                curr_config[k] = v[i]
                            else:
                                curr_config[k] = v
                                
                        params = {**common_params, **config, **curr_config}
                        
                        # 处理数据
                        if folder.name == "Q4-Part3_Align":
                            YOLO_id2cls = IDP_dla_data(**params)
                        else:
                            IDP_dla_data(**params)
                else:
                    config.update(special_config)
                    params = {**common_params, **config}
                    
                    # 处理数据
                    if folder.name == "Q4-Part3_Align":
                        YOLO_id2cls = IDP_dla_data(**params)
                    else:
                        IDP_dla_data(**params)
            else:
                params = {**common_params, **config}
                
                # 处理数据
                if folder.name == "Q4-Part3_Align":
                    YOLO_id2cls = IDP_dla_data(**params)
                else:
                    IDP_dla_data(**params)

    count_cache_files(CVAT_ROOT)
    make_yolo_cfg(YOLO_ROOT,YOLO_id2cls,output_file="DLA_Q4_20241107.yaml")

def idp_correct_pipeline(root_dir="idp_q4"):
    
    
    
    # 对齐
    TODO_ROOT = os.path.join(root_dir,"PROCESS")
    newCat = {
        1: "Paragraph",
        2: "Head",
        3: "Footnote",
        4: "HeaderFooter",
        5: "Caption",
        6: "Table",
        7: "Figure",
        8: "Formula",
        9: "Wireless",
        10: "Form"
    }
    # 遍历目录
    for folder in Path(TODO_ROOT).iterdir():
        # 检查是否为文件夹
        if folder.is_dir():
            # 获取文件夹名称
            folder_name = folder.name
            # 如果文件夹名称在预定义的列表中,则跳过
            if folder_name in ["DLA_COCO", "DLA_YOLO"]:
                continue
            
            # 处理该文件夹下的内容
            ccdata = CCTools(ROOT=str(folder))
            ccdata.update_cat(newCat=newCat)
            newObj = CCTools(ROOT=str(folder) + "_Align", CCDATA=ccdata.CCDATA, CP_IMGPATH=ccdata.ROOT.joinpath(ccdata.IMGDIR))
            newObj.save()
    
    # 合并
    
def merge_pipeline(root_dir="idp_q4/DLA_COCO",output_dir="idp_q4/DLA_COCO_MERGE"):
    # 目录下必须使用coco数据结构
    # 遍历目录
    TrainObjs =[]
    ValObjs = []
    TestObjs = []
    for folder in Path(root_dir).iterdir():
        if folder.is_dir():
            for ann in folder.joinpath("annotations").glob("*.json"):
                if "Train" in ann.stem:
                    TrainObjs.append(CCTools(ROOT=folder,ANNFILE=ann.name,IMGDIR="Train"))
                elif "Val" in ann.stem:
                    ValObjs.append(CCTools(ROOT=folder,ANNFILE=ann.name,IMGDIR="Val"))
                elif "Test" in ann.stem:
                    TestObjs.append(CCTools(ROOT=folder,ANNFILE=ann.name,IMGDIR="Test"))

    TrainObj = CCTools(ROOT=output_dir,ANNFILE="instances_Train.json",IMGDIR="images")
    TrainObj.merge(others=TrainObjs,cat_keep=True,newObj=TrainObj)
    TrainObj.save(only_ann=True)
    
    ValObj = CCTools(ROOT=output_dir,ANNFILE="instances_Val.json",IMGDIR="images")
    ValObj.merge(others=ValObjs,cat_keep=True,newObj=ValObj)
    ValObj.save(only_ann=True)
    
    TestObj = CCTools(ROOT=output_dir,ANNFILE="instances_Test.json",IMGDIR="images")
    TestObj.merge(others=TestObjs,cat_keep=True,newObj=TestObj)
    TestObj.save(only_ann=True)
    pass


if __name__ == "__main__":
    # merge()
    # filter_to_correct()
    # static()
    
    
    # pipeline_custom()
    # 临时测试，可删
    # diff_image("1.png","2.png","diff.png")
    # 正常dla流程
    # idp_dla_pipeline()
    # 纠正流程
    # idp_correct_pipeline()
    
    # 合并流程
    merge_pipeline()

    pass
    