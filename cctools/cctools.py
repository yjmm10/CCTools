from __future__ import annotations
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Union, Any,Tuple
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
            os.unlink(temp_file_name)
    
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
        
    def _get_cat(self,cat:Union[int,str]):
        """获取类别信息, 用来判断类别是否存在"""
        if isinstance(cat,int):
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
        
    def _get_img(self,img:Union[int,str]):
        """
        获取图片信息, 用来判断图片是否存在
        支持图片路径的模糊搜索
        """
        if isinstance(img,int):
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
        
        assert self.CCDATA,logger.error("CCDATA is None")
        
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
            
            if overwrite and dst_path.exists():
                os.remove(dst_path)
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
        
            
    def merge(self,other:CCTools,cat_keep:Optional[bool]=None,overwrite:Optional[bool]=None):
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
        pass
        
        
