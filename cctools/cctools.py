from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import re
import tempfile
from typing import Callable, Dict, List, Optional, Union, Any,Tuple
from pydantic import BaseModel, Field
from loguru import logger
import os, json, shutil, copy, cv2
import matplotlib.pyplot as plt

from .cc import CC


# Configure logger
def setup_logger(log_file=None):
    logger.remove()  # Remove default handler
    if log_file:
        logger.add(log_file, rotation="10 MB")  # Log to file
    else:
        logger.add(lambda msg: print(msg, end=""))  # Log to console

# Call this function at the beginning of your script
# setup_logger("log.txt")  # Uncomment to log to file
setup_logger()  # Log to console




class CCTools(BaseModel):
    """
    any object should be initialized with cocoRoot, annFile, annDir, imgDir,and support other data to process
    """
    ROOT: Optional[Path] = Path(".")
    ANNFILE: Optional[Path] = Path("instances_default.json")
    IMGDIR: Optional[Path] = Path("images")
    ANNDIR: Optional[Path] = Path("annotations")
    VISUALDIR: Optional[Path] = Path("visual")
    CCDATA: Optional[Union[dict,CC]]=None
        
    model_config = {
        "arbitrary_types_allowed": True
    }

    def __init__(self, 
                 ROOT: Optional[Union[Path, str]] = None,
                 ANNDIR: Optional[Union[Path, str]] = None,
                 IMGDIR: Optional[Union[Path, str]] = None,
                 ANNFILE: Optional[Union[Path, str]] = None,
                 VISUALDIR: Optional[Union[Path, str]] = None,
                 CCDATA: Optional[Union[dict, CC]] = None,
                 **kwargs):
        """
        初始化 CCTools 对象，确保所有路径输入都被转换为 Path 对象。
        1. 
        """
        super().__init__(**kwargs)
        self.ROOT = Path(ROOT or self.ROOT)
        self.ANNDIR = Path(ANNDIR or self.ANNDIR)
        self.IMGDIR = Path(IMGDIR or self.IMGDIR)
        self.ANNFILE = Path(ANNFILE or self.ANNFILE)
        self.VISUALDIR = Path(VISUALDIR or self.VISUALDIR)
        
        annPath = self.ROOT.joinpath(self.ANNDIR).joinpath(self.ANNFILE)
        
        exist_data = True
        if CCDATA:
            if isinstance(CCDATA, dict):
                self._CCDATA(CCDATA)
            elif isinstance(CCDATA, CC):
                self.CCDATA = CCDATA
        elif annPath.exists():
            self.CCDATA = CC(annPath)
        else:
            exist_data = False
            logger.warning(f"Annotation file not found: {annPath} and CCDATA is None")
        
        if exist_data:
            assert self.CCDATA, logger.error("CCDATA is None")
        
        # 检查路径系统完整路径
        self._checkpath()
        # 创建目录
        self._mkCCDIR(self.IMGDIR)

    def _CCDATA(self,CCDATA:Union[dict,CC])->None:
        if isinstance(CCDATA,CC):
            self.CCDATA = CCDATA
        else:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
            # Write the dictionary to a temporary JSON file
                json.dump(CCDATA, temp_file)
                temp_file_name = temp_file.name

            # Create CC object using the temporary JSON file
            self.CCDATA = CC(temp_file_name)

            # Delete the temporary file
            # os.unlink(temp_file_name)
    
    def _DICT(self)->Dict:
        logger.debug("Converting COCO object to dictionary")
        return {
            'images': list(self.CCDATA.imgs.values()),
            'annotations': list(self.CCDATA.anns.values()),
            'categories': list(self.CCDATA.cats.values())
        }
    
    def _checkpath(self,Var:Optional[str]=None)->None:
        """检查路径是否存在"""
        if Var is None:
            for Var in ["ROOT","ANNFILE","ANNDIR","IMGDIR","VISUALDIR"]:
                self._checkpath(Var)
            return
        
        if Var == "ROOT":
            if not self.ROOT.exists():
                logger.warning(f"ROOT directory not found: {self.ROOT}")
        elif Var == "ANNFILE":
            if not self.ROOT.joinpath(self.ANNDIR).joinpath(self.ANNFILE).exists():
                logger.warning(f"ANNFILE not found: {self.ANNFILE}")
        elif Var in ["ANNDIR","IMGDIR","VISUALDIR"]:
            if not self.ROOT.joinpath(eval(f"self.{Var}")).exists():
                logger.warning(f"{Var} directory not found: {self.ROOT.joinpath(eval(f'self.{Var}'))}")
        else:
            raise ValueError(f"Invalid variable: {Var}")

    def _mkCCDIR(self,DIR:Optional[Union[str,Path]],VISUALDIR:Optional[bool]=None)->None:
        """创建目录"""
        self.ROOT.mkdir(parents=True,exist_ok=True)
        self.ROOT.joinpath(DIR).mkdir(parents=True,exist_ok=True)
        self.ROOT.joinpath(self.ANNDIR).mkdir(parents=True,exist_ok=True)
        self.ROOT.joinpath(self.IMGDIR).mkdir(parents=True,exist_ok=True)
        
        if VISUALDIR:
            self.ROOT.joinpath(self.VISUALDIR).mkdir(parents=True,exist_ok=True) 
        
    def _get_cat(self,cat:Union[int,str],force_int:bool=False):
        """获取类别信息, 用来判断类别是否存在"""
        if isinstance(cat,int):
            if force_int:
                return True, cat
            cat_str = self.CCDATA.cats.get(cat,None)
            if cat_str:
                return True, cat_str.get('name',None)
            else:
                return False, None
        elif isinstance(cat,str):
            for c in self.CCDATA.cats.values():
                if c['name'] == cat:
                    return True, c.get('id',None)
            return False, None
        else:
            raise ValueError(f"Invalid category: {cat}")
        
    def _get_img(self,img:Union[int,str],force_int:bool=False):
        """
        获取图片信息, 用来判断图片是否存在
        支持图片路径的模糊搜索
        """
        if isinstance(img,int):
            if force_int:
                return True, img
            img_str = self.CCDATA.imgs.get(img,None)
            if img_str:
                return True, img_str.get('file_name',None)
            else:
                return False, None
        elif isinstance(img,str):
            for img_id in self.CCDATA.imgs.values() :
                if img_id['file_name'][:len(img)] == img:
                    return True, img_id.get('id',None)
            return False, None
        else:
            raise ValueError(f"Invalid image: {img}")
    
    def _get_imglist(self)->List[str]:
        """获取图片列表"""
        return [img['file_name'] for img in self.CCDATA.dataset['images']]
    
    def check_dir(self,VISUALDIR:Optional[bool]=None):
        """检查数据是否符合CCTools要求"""
        assert self.ROOT.exists(), f"ROOT directory not found: {self.ROOT}"
        assert self.ROOT.joinpath(self.ANNDIR).exists(), f"ANNDIR directory not found: {self.ANNDIR}"
        assert self.ROOT.joinpath(self.IMGDIR).exists(), f"IMGDIR directory not found: {self.IMGDIR}"
        assert self.ROOT.joinpath(self.ANNDIR).joinpath(self.ANNFILE).exists(), f"ANNFILE not found: {self.ANNFILE}"
        if VISUALDIR:
            assert self.ROOT.joinpath(self.VISUALDIR).exists(), f"VISUALDIR directory not found: {self.VISUALDIR}"

    def visual(self,overwrite:bool=True):
        """可视化数据"""
        
        visual_dir = self.ROOT.joinpath(self.VISUALDIR)
        if overwrite and visual_dir.exists():
            shutil.rmtree(visual_dir)
        visual_dir.mkdir(parents=True,exist_ok=True)
        
        self.check_dir(VISUALDIR=True)
        
        for img_id in self.CCDATA.getImgIds() :
            file_name = self.CCDATA.imgs[img_id]['file_name']
            img_path = self.ROOT.joinpath(self.IMGDIR).joinpath(file_name)
            out_path = self.ROOT.joinpath(self.VISUALDIR).joinpath(file_name)
            
            # 获取该图片所有的anno
            anno_ids = self.CCDATA.getAnnIds(imgIds=img_id)
            one_anns = [self.CCDATA.anns[i] for i in anno_ids]
            
            image = cv2.imread(img_path)
            plt.imshow(image) 
            plt.axis('off')
            self.CCDATA.showBBox(anns=one_anns)
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
        logger.info(f"Visualization completed. Output saved to {self.ROOT.joinpath(self.VISUALDIR)}")
        
    def save(self,New:CCTools,visual:bool=True,overwrite:bool=True): 
        # 自己保存自己，存在数据无法保存问题
        assert self.CCDATA, logger.error("CCDATA is None")
        
        CCDIT = self._DICT()

        src_annFile,src_root,src_annDir,src_imgDir = self.ANNFILE,self.ROOT,self.ANNDIR,self.IMGDIR
        if New.ANNFILE is None:
            dst_annFile,dst_root,dst_annDir,dst_imgDir = src_annFile,src_root,src_annDir,src_imgDir
        else:
            dst_annFile,dst_root,dst_annDir,dst_imgDir = New.ANNFILE,New.ROOT,New.ANNDIR,New.IMGDIR
        
        with open(dst_root.joinpath(dst_annDir).joinpath(dst_annFile), 'w', encoding='utf-8') as json_file:
            json.dump(CCDIT, json_file, ensure_ascii=False, indent=2)  # Use indent parameter to beautify output
        
        # 图片处理
         # Copy images to new path
        for img in CCDIT['images']:
            src_path = src_root.joinpath(src_imgDir,img['file_name'])
            dst_path = dst_root.joinpath(dst_imgDir,img['file_name'])
            
            # if overwrite and dst_path.exists():
            #     os.remove(dst_path)
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
        New.CCDATA = self.CCDATA
        # 保存
        if visual:
            New.visual(overwrite=overwrite)
        
    def static(self)->Dict:
        """统计数据"""
        stats = {
                "total_images": len(self.CCDATA.dataset['images']),
                "total_annotations": len(self.CCDATA.dataset['annotations']),
                "annotations_per_category": {},
                "annotations_per_image": {},
                "images_without_annotations": 0,
                "imgId_without_annotations": []
            }

        # Calculate annotations per category
        for cat in self.CCDATA.dataset['categories']:
            ann_count = len(self.CCDATA.getAnnIds(catIds=[cat['id']]))
            stats["annotations_per_category"][cat['name']] = ann_count

        # Calculate annotations per image
        for img_id in self.CCDATA.dataset['images']:
            ann_count = len(self.CCDATA.getAnnIds(imgIds=[img_id['id']]))
            stats["annotations_per_image"][img_id['file_name']] = ann_count
            if ann_count == 0:
                stats["imgId_without_annotations"].append(img_id['id'])

        logger.info("Dataset statistics calculation completed successfully.")
        return stats
    
    def update_cat(self, newCat:Dict[int,str]):
        """
        根据给定的类别字典更新数据集中的类别
        """
        dataset = self.CCDATA.dataset
        # Check if categories need to be expanded
        rawCat = {cat['id']: cat['name'] for cat in dataset['categories']}
        rawId = max(rawCat.keys())
        
        for _, new_name in newCat.items():
            if new_name not in rawCat.values():
                rawId += 1
                dataset['categories'].append({
                    'id': rawId,
                    'name': new_name,
                    'supercategory': ''
                })
                logger.info(f"Added new category: ID {rawId}, name '{new_name}'")
        
        newCat = {cat['id']: new_id for cat in dataset['categories'] for new_id, name in newCat.items() if cat['name'] == name}
            
        # Update categories
        for category in dataset['categories']:
            if category['id'] in newCat.keys():
                old_id = category['id']
                category['id'] = newCat[old_id]
                logger.debug(f"Updating category ID: {old_id} -> {category['id']}")

        # Update annotations
        for annotation in dataset['annotations']:
            if annotation['category_id'] in newCat.keys():
                old_id = annotation['category_id']
                annotation['category_id'] = newCat[old_id]
                logger.debug(f"Updating annotation category ID: {old_id} -> {annotation['category_id']}")
        self.CCDATA.createIndex()
        
    def rename_cat(self, old_name:str, new_name:str)->None:
        """
        修改类别名称
        """
        dataset = self.CCDATA.dataset
        for category in dataset['categories']:
            if category['name'] == old_name:
                category['name'] = new_name
                logger.debug(f"Updating category name: {old_name} -> {new_name}")
        self.CCDATA.createIndex()
    
    def _align_cat(self,otherCat:Dict,cat_keep:bool=True)->Dict:
        """
        对齐两个数据集的类别
        """
        rawCat = {cat['id']: cat['name'] for cat in self.CCDATA.dataset['categories']}
        if cat_keep:
            newCat = rawCat
            maxId = len(rawCat.keys())
            for cat in otherCat:
                if cat not in rawCat:
                    maxId += 1
                    rawCat[maxId] = otherCat[cat]
                else:
                    logger.warning(f"Category {cat} already exists in self dataset")
        else:
            newCat = otherCat
            maxId = len(rawCat.keys())
            for cat in rawCat:
                if cat not in otherCat:
                    maxId += 1
                    newCat[maxId] = rawCat[cat]
                else:
                    logger.warning(f"Category {cat} already exists in other dataset")
        
        return newCat        
    
    def _updateIndex(self,imgIndex:Optional[int]=None,annIndex:Optional[int]=None):
        """
        更新数据集索引
        """

        imgIndex = imgIndex or 1
        if imgIndex < 1:
            raise ValueError("imgIndex must be greater than 0")
        annIndex = annIndex or 1
        if annIndex < 1:
            raise ValueError("annIndex must be greater than 0")

        img_id_map = {}
        for img in self.CCDATA.dataset['images']:
            img_id_map[img['id']] = imgIndex
            img['id'] = imgIndex
            imgIndex += 1
            
        for img_ann in self.CCDATA.dataset['annotations']:
            if img_ann['image_id'] in img_id_map:
                img_ann['image_id'] = img_id_map[img_ann['image_id']]
            else:
                logger.warning(f"Image ID {img_ann['image_id']} not found in img_id_map")


        ann_id_map = {} # old: new                
        for anns in self.CCDATA.dataset['annotations']:
            ann_id_map[anns['id']] = annIndex
            anns['id'] = annIndex
            annIndex += 1
        
        self.CCDATA.createIndex()
        
            
    def _merge(self,other:CCTools,cat_keep:Optional[bool]=None,overwrite:Optional[bool]=None):
        """
        合并两个数据集
        cat_keep: 类别保留方式, True: 以self为主, False: 以other为主
        overwrite: 图片名称存在是否覆盖
        """
        cat_keep = cat_keep or True
        overwrite = overwrite or False
        otherCat = {cat['id']: cat['name'] for cat in other.CCDATA.dataset['categories']}
        newCat = self._align_cat(otherCat=otherCat,cat_keep=cat_keep)
        raw_imglist = self._get_imglist()
        other_imgidlist = [img['id'] for img in other.CCDATA.dataset['images']]
        
        # 对齐两个数据集的类别
        self.update_cat(newCat=newCat)
        other.update_cat(newCat=newCat)
        
        if cat_keep:
            self._updateIndex(imgIndex=1,annIndex=1)
            other._updateIndex(imgIndex=len(self.CCDATA.imgs)+1,annIndex=len(self.CCDATA.anns)+1)
        else:
            other._updateIndex(imgIndex=1,annIndex=1)
            self._updateIndex(imgIndex=len(other.CCDATA.imgs)+1,annIndex=len(other.CCDATA.anns)+1)
        
        
        for img in other.CCDATA.dataset['images']:
            # 检查是否重复
            if img['file_name'] in raw_imglist:
                if overwrite:
                    self.CCDATA.dataset['images'].append(img)
                else:
                    logger.warning(f"Image {img['file_name']} already exists in other dataset")
            else:
                self.CCDATA.dataset['images'].append(img)


        for ann in other.CCDATA.dataset['annotations']:
            # 检查是否重复
            if ann['image_id'] in other_imgidlist:
                self.CCDATA.dataset['annotations'].append(ann)

        self.CCDATA.createIndex()

    def merge(self,others:Union[CCTools,List[CCTools]],cat_keep:Optional[bool]=None,overwrite:Optional[bool]=None):
        """
        合并两个数据集
        cat_keep: 类别保留方式, True: 以self为主, False: 以other为主
        overwrite: 图片名称存在是否覆盖
        """
        if isinstance(others,CCTools):
            others = [others]
        for other in others:
            self._merge(other=other,cat_keep=cat_keep,overwrite=overwrite)
        

    def _filter(self, catIds: Optional[List[int]] = [], imgIds: Optional[List[int]] = [], annIds: Optional[List[int]] = [], mod:Optional[str]="and"):
        """
        过滤数据集
        mod: 过滤方式, "and": 同时满足, "or": 满足其一的并集
        """
        if mod == "and":
            return self._filter_and(catIds=catIds,imgIds=imgIds,annIds=annIds)
        elif mod == "or":
            return self._filter_or(catIds=catIds,imgIds=imgIds,annIds=annIds)
        else:
            raise ValueError(f"Invalid mod: {mod}")

    
    def _filter_and(self, catIds: Optional[List[int]] = [], imgIds: Optional[List[int]] = [], annIds: Optional[List[int]] = []):
        """
        为空，表示所有值
        """
        final_annIds = []
            
        annLists = self.CCDATA.getAnnIds(imgIds=imgIds,catIds=catIds)
        if annIds:
            for ann in annLists:
                if ann in annIds:
                    final_annIds.append(ann)
        else:
            final_annIds = annLists
            
        return final_annIds
    
    def _filter_or(self, catIds: Optional[List[int]] = [], imgIds: Optional[List[int]] = [], annIds: Optional[List[int]] = []):
        """
        为空，表示不获取任何值
        """
        res_cats = self._filter_and(catIds=catIds) if catIds else []
        res_imgs = self._filter_and(imgIds=imgIds) if imgIds else []
        res_anns = self._filter_and(annIds=annIds) if annIds else []
        # 计算并集
        final_annIds = set(res_cats + res_imgs + res_anns)
        return list(final_annIds)
    
    def _get_imgIds_by_annIds(self,annIds:List[int]):
        """
        根据annIds获取图片id
        """
        return list(set([ann['image_id'] for ann in self.CCDATA.dataset['annotations'] if ann['id'] in annIds]))
    
    def _get_catIds_by_annIds(self,annIds:List[int]):
        """
        根据annIds获取类别id
        """
        return list(set([ann['category_id'] for ann in self.CCDATA.dataset['annotations'] if ann['id'] in annIds]))
    
    def _get_data(self,annIds:List[int],level:str="img"):
        """
        根据annIds获取数据
        """
        if not annIds:
            return [],[],[]
        
        if level == "img":
            res_imgIds = self._get_imgIds_by_annIds(annIds=annIds)
            res_annIds = self.CCDATA.getAnnIds(imgIds=res_imgIds)
            res_catIds = self._get_catIds_by_annIds(annIds=res_annIds)
        elif level == "ann":
            res_imgIds = self._get_imgIds_by_annIds(annIds=annIds)
            res_annIds = annIds
            res_catIds = self._get_catIds_by_annIds(annIds=res_annIds)
        else:
            raise ValueError(f"Invalid level: {level}")
        
        return res_catIds,res_imgIds,res_annIds
    
    def _gen_dict(self,catIds:List[int],imgIds:List[int],annIds:List[int],alignCat:bool=True):
        if not catIds:
            return {}
        new_dataset = {
            'info': self.CCDATA.dataset['info'],
            'licenses': self.CCDATA.dataset['licenses'],
            'images': [img for img in self.CCDATA.dataset['images'] if img['id'] in imgIds],
            'annotations': [ann for ann in self.CCDATA.dataset['annotations'] if ann['id'] in annIds],
            'categories': [cat for cat in self.CCDATA.dataset['categories'] if cat['id'] in catIds]
        }
        if isinstance(alignCat,bool):
            new_dataset['categories'] = [cat for cat in self.CCDATA.dataset['categories']]

        return new_dataset
     
    def filter(self, cats: Optional[List[Union[int,str]]] = [], imgs: Optional[List[Union[int,str]]] = [], annIds: Optional[List[int]] = [], mod:Optional[str]="and", newObj:Optional[CCTools]=None,visual:bool=False,alignCat:bool=True,sep_data:bool=False,level="img")->Union[dict,CCTools]:
        """
        过滤数据集,图像支持模糊搜索，类别支持id或名称搜索，标签为过滤的结果，设置alignCat对齐类别结果
        mod: 过滤方式, "and": 同时满足, "or": 满足单个的所有并集
        level: 过滤结果的级别, "img": 图片, "ann": 标签，即过滤后的标注数量
        sep_data: 是否分离数据，即是否返回新的CCTools对象
        alignCat: 对齐类别，bool: True, 对齐self, Dict: 对齐指定的类别
        """
        # 获取id
        imgIds = [self._get_img(img,force_int=True)[1] for img in imgs if self._get_img(img,force_int=True)[0]]
        catIds = [self._get_cat(cat,force_int=True)[1] for cat in cats if self._get_cat(cat,force_int=True)[0]]
        
        
        dst_annIds = self._filter(catIds=catIds, imgIds=imgIds, annIds=annIds,mod=mod)
        dst_catIds,dst_imgIds,dst_annIds = self._get_data(annIds=dst_annIds,level=level)
        dst_dict = self._gen_dict(catIds=dst_catIds,imgIds=dst_imgIds,annIds=dst_annIds,alignCat=alignCat)
        
        
        if sep_data:
            all_annIds = [ann['id'] for ann in self.CCDATA.dataset['annotations']] 
            sep_annIds = set(all_annIds) - set(dst_annIds)
            sep_catIds,sep_imgIds,sep_annIds = self._get_data(annIds=sep_annIds,level=level)
            sep_dict = self._gen_dict(catIds=sep_catIds,imgIds=sep_imgIds,annIds=sep_annIds,alignCat=alignCat)
            if sep_dict:
                self._CCDATA(sep_dict)  
            else:
                logger.warning("No data after separation,keep the original data")
            
        if newObj:
            newObj._CCDATA(dst_dict)
            imglist = newObj._get_imglist()
            srcRoot = self.ROOT.joinpath(self.IMGDIR)
            dstRoot = newObj.ROOT.joinpath(newObj.IMGDIR)
            for img in imglist:
                src_path = srcRoot.joinpath(img)
                dst_path = dstRoot.joinpath(img)
                shutil.copy2(src_path, dst_path)
            newObj.save(New=newObj,visual=visual)
            return newObj
        else:
            return dst_dict
        
    def correct(self, api_url:Callable, cats:Union[int,str,list], newObj:Optional[CCTools]=None, visual:bool=False):
        if isinstance(cats,list):
            catIds = [self._get_cat(cat,force_int=True)[1] for cat in cats]
        else:
            catIds = [self._get_cat(cats,force_int=True)[1]]
             
        imgDir = self.ROOT.joinpath(self.IMGDIR)
        
        newCC = copy.deepcopy(self.CCDATA)
        sys_anns = newCC.getCatIds()
        imgIds = newCC.getImgIds()

        
        for img_id in imgIds:
            img = newCC.loadImgs(img_id)[0]
            imgpath = imgDir.joinpath(img['file_name'])

            one_annoIds = newCC.getAnnIds(imgIds=img['id'])
            one_anns = newCC.loadAnns(one_annoIds)
            for ann in one_anns:                
                if ann['category_id'] not in catIds:
                    continue
                
                bbox = ann['bbox']
                x, y, w, h = bbox
                x1, y1 = int(x), int(y)
                x2, y2 = int(x+w), int(y+h)
                bbox = [x1,y1,x2,y2]

                # Call API, pass in the temporary file path
                api_res = api_url(imgpath,bbox)
                if api_res not in sys_anns:
                    raise ValueError(f"API returned category ID is not in system categories, please confirm the value is between 1 and {len(sys_anns)}")
                # Apply API results directly to annotation
                ann['category_id'] = api_res

        if newObj:
            newObj.CCDATA=newCC
            imglist = newObj._get_imglist()
            srcRoot = self.ROOT.joinpath(self.IMGDIR)
            dstRoot = newObj.ROOT.joinpath(newObj.IMGDIR)
            for img in imglist:
                src_path = srcRoot.joinpath(img)
                dst_path = dstRoot.joinpath(img)
                shutil.copy2(src_path, dst_path)
            newObj.save(New=newObj,visual=visual)
            return newObj
        else:
            self.CCDATA=newCC
            
    def split(self,ratio:List[float]=[0.7,0.2,0.1],by_file=False):
        """
        ratio:划分比例，按照训练集、验证集、测试集顺序
        by_file:是否按照文件划分

        """
        if len(ratio)==2:
            ratio.append(1-sum(ratio))

        imglists = self._get_imglist()
        samebooks = defaultdict(list)
        for image in imglists:
            match = re.match(r'(.+?)_(\d+)\..*', image)
            if match:
                prefix, page = match.groups()
                samebooks[prefix].append(image)
        samebooks = dict(samebooks)
