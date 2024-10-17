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
        
   
        