from collections import defaultdict
import copy
import json
from pathlib import Path
from cctools.cctools import CCTools

from .logger import logger

class Pipeline:
    def __init__(self,):
        pass
    
    def split_data(self,config:dict={'cctools':None,
                                     'ratio':[0.7,0.2,0.1],
                                     'by_file':False,
                                     'visual':False,
                                     'newObj':None}
                   ):
        """
        固定划分数据集，根据缓存来处理
        cache_path: 当存在newObj时，cache_path为newObj的ROOT，否则为cctools的ROOT
        """
        NAME = "split_data"
        cctools = config.get('cctools',None)
        ratio = config.get('ratio',[0.7,0.2,0.1])
        by_file = config.get('by_file',False)
        visual = config.get('visual',False)
        newObj = config.get('newObj',None)
        

        
        assert cctools, logger.error("CCTools is None")      
        # assert newObj, logger.error("NewObj is None")

        cache_path = cctools.ROOT
        
        
        cache_file = Path(NAME + f"_{cctools.ROOT.stem}_img{len(cctools.CCDATA.dataset['images'])}_ratio{''.join([str(int(r*10)) for r in ratio])}{'_byfile' if by_file else ''}.cache")
        cache_file =cache_path.joinpath(cache_file)
        # 创建缓存目录
        cache_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache file: {cache_file}")
        
        # 检查缓存文件是否存在
        if cache_file.exists():
            logger.info(f"Found cache file: {cache_file}")
            with open(cache_file, "r") as f:
                cache_data = json.load(f)
                
            # 对缓存进行处理
            # 1. 获取缓存对应的文件信息
            # 2. 划分数据集
            # 3. 保存为json文件
            trainObj = CCTools(CCTOOLS_OBJ=newObj,IMGDIR="Train",ANNFILE="instances_Train.json")
            cctools.filter(imgs=cache_data["train"],newObj=trainObj,visual=visual,mod="and",alignCat=True,sep_data=False,level="img")
            trainObj.save(visual=visual)

            valObj = CCTools(CCTOOLS_OBJ=newObj,IMGDIR="Val",ANNFILE="instances_Val.json")  
            cctools.filter(imgs=cache_data["val"],newObj=valObj,visual=visual,mod="and",alignCat=True,sep_data=False,level="img")
            valObj.save(visual=visual)
            
            testObj = CCTools(CCTOOLS_OBJ=newObj,IMGDIR="Test",ANNFILE="instances_Test.json")
            cctools.filter(imgs=cache_data["test"],newObj=testObj,visual=visual,mod="and",alignCat=True,sep_data=False,level="img")
            testObj.save(visual=visual)
            
                
        else:
            # 制作缓存
            # 1. 读取CCTool对象
            # 2. 划分数据集
            # 3. 保存为json文件
            logger.info(f"Cache file not found: {cache_file}")
            
            trainObj,valObj,testObj = cctools.split(ratio=ratio,newObj=newObj,by_file=by_file,visual=visual,merge=False)
            # visual存在时会自动保存数据
            if not visual:
                trainObj.save(visual=visual)
                valObj.save(visual=visual)
                testObj.save(visual=visual)
            
            
            train_img = trainObj._get_imglist()
            val_img = valObj._get_imglist()
            test_img = testObj._get_imglist()
            
            cache_data = {
                "train":train_img,
                "val":val_img,
                "test":test_img
            }
            Path(cache_path).mkdir(parents=True,exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
            
            logger.info(f"Save cache file: {cache_file}")
            
    
    def static_data(self,root,static_file="static.csv"):
        """
        判断目录下是否存在特定结构,存在则获取该结构所在目录
        
        Args:
            root: 根目录路径
            
        Returns:
            list: 包含特定结构的目录列表
        """
        root = Path(root)
        target_dirs = ["images", "annotations", "Train", "Test","Val"]
        # 图片目录对应的标注文件，仅支持一一对应
        data_match = {"images":"instances_default.json",
                      "Train":"instances_Train.json",
                      "Val":"instances_Val.json",
                      "Test":"instances_Test.json"}
        result = []
        # 遍历根目录下所有子目录,查找所有的coco目录
        for path in root.rglob("*"):
            if path.is_dir():
                # 检查当前目录名是否在目标目录列表中
                if path.name in target_dirs:
                    result.append(str(path.parent))
        coco_results = list(set(result)) if result else None     
    
        
        all_result = {}
        for coco_dir in coco_results:
            # 判断目录下是否包含data_match中的目录结构
            coco_root = Path(coco_dir)
            IMGDIR,ANNFILE = None,None
            for k,v in data_match.items():
                if coco_root.joinpath(k).exists() and coco_root.joinpath("annotations",v).exists():
                    IMGDIR,ANNFILE = k,v

            CCTOOLS_OBJ = CCTools(ROOT=coco_root,IMGDIR=IMGDIR,ANNFILE=ANNFILE)
            result = CCTOOLS_OBJ.static()
            result.pop("images_without_annotations")
            result.pop("imgId_without_annotations")
            logger.info(f"Static result: {coco_dir}, result: {result}")
            
            all_result[coco_root.stem+f"--{ANNFILE.split('.')[0]}"] = result
            
        # 统计所有数据集的总数
        total_stats = {
            'total_images': 0,
            'total_annotations': 0,
            'annotations_per_category': defaultdict(int),
            'annotations_per_image': {}
        }
        
        # 遍历每个数据集的统计结果
        for dataset_name, stats in all_result.items():
            # 累加图片和标注总数
            total_stats['total_images'] += stats['total_images'] 
            total_stats['total_annotations'] += stats['total_annotations']
            
            # 累加每个类别的标注数量
            for cat, count in stats['annotations_per_category'].items():
                total_stats['annotations_per_category'][cat] += count
                
            # 合并每张图片的标注数量
            total_stats['annotations_per_image'].update(stats['annotations_per_image'])
            
        # 将defaultdict转换为普通dict
        total_stats['annotations_per_category'] = dict(total_stats['annotations_per_category'])
        
        # 添加总统计结果到all_result
        all_result['total'] = total_stats
        
        logger.info(f"Total statistics: {total_stats}")
        
        # 创建Excel文件保存统计结果
        import pandas as pd
        
        # 创建一个Excel writer对象
        with pd.ExcelWriter('static_file.xlsx') as writer:
            # Sheet1: 总体统计
            total_rows = []
            # 添加总体统计行
            total_rows.append({
                'Dataset': 'Total',
                'Images': total_stats['total_images'],
                'Annotations': total_stats['total_annotations']
            })
            # 添加每个数据集的统计行
            for dataset_name, stats in all_result.items():
                if dataset_name != 'total':
                    total_rows.append({
                        'Dataset': dataset_name,
                        'Images': stats['total_images'],
                        'Annotations': stats['total_annotations']
                    })
            total_df = pd.DataFrame(total_rows)
            total_df.to_excel(writer, sheet_name='总体统计', index=False)
            
            # Sheet2: 每个类别的标注数量
            # 创建一个字典来存储每个数据集的类别统计
            category_stats = {}
            # 添加总体统计
            category_stats['Total'] = total_stats['annotations_per_category']
            # 添加每个数据集的统计
            for dataset_name, stats in all_result.items():
                if dataset_name != 'total':
                    category_stats[dataset_name] = stats['annotations_per_category']
            
            # 转换为DataFrame并转置
            category_df = pd.DataFrame(category_stats).T
            # 填充缺失值为0
            category_df = category_df.fillna(0)
            category_df.to_excel(writer, sheet_name='类别统计')
            
            # Sheet3: 每张图片的标注数量
            image_stats = []
            current_dataset = None
            for dataset_name, stats in all_result.items():
                if dataset_name != 'total':
                    current_dataset = dataset_name
                    for i,(img, ann_count) in enumerate(stats['annotations_per_image'].items()):
                        if i == 0:  
                            image_stats.append({
                                'Dataset': current_dataset,
                                'Image': img,
                                'Annotations': ann_count
                            })
                        else:
                            image_stats.append({
                                'Dataset': None,
                                'Image': img,
                                'Annotations': ann_count
                            })
            
            image_df = pd.DataFrame(image_stats)
            image_df.to_excel(writer, sheet_name='图片标注统计', index=False)
        
        return all_result,total_stats

        
    
    
    def run(self,pipeline_name:str,pipeline_config:dict):
        pass