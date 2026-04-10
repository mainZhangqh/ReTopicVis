#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask 后端
- 旧主题 / 新主题层次树、气泡图、桑基图等接口
- 新增 /api/topic_similarity 主题相似度映射接口
- 新增 AI 对话接口
- 修复机构统计接口：添加机构名称规范化，支持模糊匹配
"""

from __future__ import annotations

import json
import logging
import os
import traceback
import re
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
from datetime import datetime

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.cluster.hierarchy import to_tree as scipy_to_tree  # 避免重名
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import math

from sklearn.feature_extraction.text import TfidfVectorizer

# -------------------- AI 相关导入 -------------------- #
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers 库未安装，AI功能将不可用")

from dataclasses import dataclass

# -------------------- 基础配置 -------------------- #
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

DATA_DIR: Path = Path(__file__).with_name("data")  # 兼容 py 脚本位置
# 修复：在文件顶部定义 DATA_FILE
DATA_FILE = DATA_DIR / "final_docs_data.json"

SIM_FILE_OLD = os.path.join(DATA_DIR, "topic_similarity_qwen_old.json")
SIM_FILE_NEW = os.path.join(DATA_DIR, "topic_similarity_qwen_new.json")

# AI 模型路径
MODEL_PATH = r"D:\MODELS\LLM-Research\Qwen2-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" if TRANSFORMERS_AVAILABLE else "cpu"

# -------------------- 日志 -------------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- 机构名称规范化配置 -------------------- #
# 机构同义词映射表（可根据实际情况扩展）
ORG_SYNONYMS = {
    # 清华大学
    'tsinghua university': [
        'tsinghua', 'tsing hua', '清华', '清华大学', 
        'tsinghua univ', 'qinghua', 'thu'
    ],
    # 北京大学
    'peking university': [
        'pku', 'peking univ', '北京大学', '北京大學',
        'peking', 'beijing university'
    ],
    # 麻省理工学院
    'massachusetts institute of technology': [
        'mit', 'massachusetts inst tech', '麻省理工',
        '麻省理工学院'
    ],
    # 斯坦福大学
    'stanford university': [
        'stanford', 'stanford univ', '斯坦福', '斯坦福大学'
    ],
    # 卡内基梅隆大学
    'carnegie mellon university': [
        'cmu', 'carnegie mellon', '卡内基梅隆', '卡内基梅隆大学'
    ],
    # 加州大学伯克利分校
    'university of california berkeley': [
        'uc berkeley', 'berkeley', '加州伯克利', '加州大学伯克利分校',
        'univ calif berkeley'
    ],
    # 中国科学院
    'chinese academy of sciences': [
        'cas', '中国科学院', '中科院',
        'chinese acad sci'
    ],
    # 微软
    'microsoft': [
        'microsoft research', 'microsoft corp', '微软',
        'msr', 'microsoft corporation'
    ],
    # 谷歌
    'google': [
        'google inc', 'google llc', '谷歌',
        'google research', 'google ai'
    ],
    # 百度
    'baidu': [
        'baidu inc', '百度', '百度公司',
        'baidu research'
    ],
    # 腾讯
    'tencent': [
        'tencent inc', '腾讯', '腾讯公司',
        'tencent research'
    ],
    # 阿里巴巴
    'alibaba': [
        'alibaba group', 'alibaba inc', '阿里巴巴',
        'alibaba research'
    ],
    # 华为
    'huawei': [
        'huawei technologies', '华为', '华为技术有限公司',
        'huawei research'
    ],
    # 牛津大学
    'university of oxford': [
        'oxford', 'oxford university', '牛津', '牛津大学'
    ],
    # 剑桥大学
    'university of cambridge': [
        'cambridge', 'cambridge university', '剑桥', '剑桥大学'
    ],
    # 帝国理工学院
    'imperial college london': [
        'imperial college', 'imperial', '帝国理工',
        '帝国理工学院'
    ],
    # 华盛顿大学
    'university of washington': [
        'uw', 'uwashington', '华盛顿大学', 'univ washington'
    ],
    # 康奈尔大学
    'cornell university': [
        'cornell', '康奈尔', '康奈尔大学'
    ],
    # 密歇根大学
    'university of michigan': [
        'umich', 'michigan', '密歇根大学'
    ],
    # 宾夕法尼亚大学
    'university of pennsylvania': [
        'upenn', 'penn', '宾夕法尼亚大学'
    ],
    # 哥伦比亚大学
    'columbia university': [
        'columbia', '哥伦比亚大学'
    ],
    # 耶鲁大学
    'yale university': [
        'yale', '耶鲁', '耶鲁大学'
    ],
    # 普林斯顿大学
    'princeton university': [
        'princeton', '普林斯顿', '普林斯顿大学'
    ],
    # 加州理工学院
    'california institute of technology': [
        'caltech', '加州理工', '加州理工学院'
    ],
    # 南洋理工大学
    'nanyang technological university': [
        'ntu', '南洋理工', '南洋理工大学'
    ],
    # 香港大学
    'university of hong kong': [
        'hku', '香港大学'
    ],
    # 香港中文大学
    'chinese university of hong kong': [
        'cuhk', '香港中文大学'
    ],
    # 东京大学
    'university of tokyo': [
        'tokyo university', '东京大学', '東大', 'utokyo'
    ],
    # 京都大学
    'kyoto university': [
        'kyoto', '京都大学'
    ],
    # 首尔大学
    'seoul national university': [
        'snu', '首尔大学', '서울대학교'
    ],
    # 新加坡国立大学
    'national university of singapore': [
        'nus', '新加坡国立大学'
    ],
    # 苏黎世联邦理工学院
    'eth zurich': [
        'eth', 'eth zürich', '苏黎世联邦理工', 
        '联邦理工学院', '瑞士联邦理工学院'
    ],
    # 洛桑联邦理工学院
    'epfl': [
        'ecole polytechnique fédérale de lausanne',
        '洛桑联邦理工', '联邦理工学院洛桑'
    ]
}

def normalize_org_name(org_name):
    """
    规范化机构名称，用于模糊匹配
    
    参数:
        org_name: 原始机构名称
    
    返回:
        标准化后的机构名称
    """
    if not org_name:
        return ""
    
    # 确保是字符串
    if not isinstance(org_name, str):
        try:
            org_name = str(org_name)
        except:
            return ""
    
    # 转换为小写并去除首尾空格
    normalized = org_name.lower().strip()
    
    # 如果为空，返回空字符串
    if not normalized:
        return ""
    
    # 去除常见的标点符号和特殊字符
    normalized = re.sub(r'[^\w\s\-&]', ' ', normalized)
    
    # 替换常见的缩写和变体
    replacements = {
        'univ': 'university',
        'inst': 'institute',
        'dept': 'department',
        'lab': 'laboratory',
        'corp': 'corporation',
        'inc': '',
        'ltd': '',
        'co': '',
        'llc': '',
        '&': 'and',
        'technol': 'technology',
        'sci': 'science',
        'engn': 'engineering',
        'res': 'research',
        'acad': 'academy',
        'natl': 'national',
        'int': 'international',
        'info': 'information',
        'comp': 'computer',
        'elec': 'electrical',
        'mech': 'mechanical',
        'chem': 'chemical',
        'phys': 'physics',
        'math': 'mathematics',
        'bio': 'biology',
        'med': 'medical',
        'hosp': 'hospital',
        'ctr': 'center',
        'ctr': 'centre',
        'sch': 'school',
        'coll': 'college',
        'dept': 'department',
        'div': 'division',
        'grp': 'group',
        'tech': 'technology',
        'telecom': 'telecommunications',
        'comm': 'communications',
        'sys': 'systems',
        'dev': 'development',
        'mgr': 'manager',
        'mgt': 'management',
        'admin': 'administration',
        'assoc': 'association',
        'soc': 'society',
        'org': 'organization',
        'std': 'standard',
        'proc': 'processing',
        'ctrl': 'control',
        'algo': 'algorithm',
        'arch': 'architecture',
        'auto': 'automation',
        'anal': 'analysis',
        'appl': 'applied',
        'artif': 'artificial',
        'biomed': 'biomedical',
        'commun': 'communications',
        'comput': 'computer',
        'constr': 'construction',
        'distrib': 'distribution',
        'environ': 'environmental',
        'exper': 'experimental',
        'indust': 'industrial',
        'informat': 'information',
        'intell': 'intelligence',
        'manufact': 'manufacturing',
        'mater': 'materials',
        'method': 'methodology',
        'multimedia': 'multimedia',
        'opt': 'optical',
        'organ': 'organizational',
        'perform': 'performance',
        'polit': 'political',
        'pract': 'practice',
        'pres': 'president',
        'prod': 'production',
        'prog': 'programming',
        'qual': 'quality',
        'quant': 'quantitative',
        'reliab': 'reliability',
        'rep': 'representative',
        'saf': 'safety',
        'sat': 'satellite',
        'sec': 'security',
        'serv': 'service',
        'simul': 'simulation',
        'sol': 'solution',
        'spec': 'special',
        'stat': 'statistical',
        'strat': 'strategic',
        'struct': 'structural',
        'super': 'supervisory',
        'supp': 'support',
        'sustain': 'sustainable',
        'synth': 'synthetic',
        'tech': 'technical',
        'tele': 'telecommunications',
        'test': 'testing',
        'theor': 'theoretical',
        'trad': 'traditional',
        'trans': 'transmission',
        'util': 'utility',
        'valid': 'validation',
        'var': 'variable',
        'verif': 'verification',
        'vers': 'version',
        'vis': 'visual',
        'vol': 'volume',
        'wireless': 'wireless',
        'work': 'working',
        'worldwide': 'world wide'
    }
    
    # 替换缩写
    for short, full in replacements.items():
        # 使用单词边界来确保只替换完整的单词
        normalized = re.sub(rf'\b{short}\b', full, normalized)
    
    # 移除多余的空格
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # 移除常见的后缀（按顺序处理）
    suffixes = [
        # 英文后缀
        'university', 'college', 'institute', 'school', 
        'department', 'laboratory', 'lab', 'center', 'centre',
        'corporation', 'company', 'limited', 'inc',
        'research', 'development', 'technologies', 'technology',
        'sciences', 'science', 'engineering', 'engineers',
        'academy', 'association', 'society', 'organization',
        'group', 'holding', 'holdings', 'international',
        'national', 'regional', 'global', 'worldwide',
        'bureau', 'agency', 'administration', 'authority',
        'division', 'section', 'unit', 'team',
        'network', 'system', 'systems', 'solution', 'solutions',
        'service', 'services', 'consulting', 'consultants',
        
        # 中文后缀（英文翻译）
        'univ', 'inst', 'dept', 'tech',
        
        # 通用后缀
        'the', 'of', 'and', 'for', 'on', 'in', 'at', 'by',
        'to', 'from', 'with', 'as', 'a', 'an', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'having', 'do', 'does', 'did', 'doing'
    ]
    
    # 移除后缀
    for suffix in suffixes:
        if normalized.endswith(f' {suffix}'):
            normalized = normalized[:-len(f' {suffix}')].strip()
        elif normalized == suffix:
            normalized = ''
    
    # 再次清理多余空格
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def unify_org_name(org_name):
    """
    统一机构名称，将同义词映射到标准名称
    
    参数:
        org_name: 原始机构名称
    
    返回:
        统一后的标准机构名称
    """
    if not org_name:
        return org_name
    
    normalized = normalize_org_name(org_name)
    
    if not normalized:
        return org_name
    
    # 检查是否匹配同义词表中的任何变体
    for standard_name, synonyms in ORG_SYNONYMS.items():
        # 检查标准化后的名称是否完全匹配标准名称
        if normalized == standard_name:
            return standard_name
        
        # 检查标准化后的名称是否包含标准名称
        if standard_name in normalized:
            return standard_name
        
        # 检查标准化后的名称是否匹配任何同义词
        for synonym in synonyms:
            if synonym == normalized or synonym in normalized:
                return standard_name
    
    # 如果没有匹配，返回标准化后的名称
    return normalized

# -------------------- 全局数据 -------------------- #
_data_cache = {
    "papers_df": None,
    "topics_info_df": None,
    "topic_keywords_df": None,
    "similarity_matrix": None,
    "topic_summaries": None
}

# -------------------- AI 模型管理 -------------------- #
@dataclass
class TopicAnalysisResult:
    """主题分析结果"""
    topic_id: int
    topic_name: str
    summary: str
    key_concepts: List[str]
    research_trends: List[str]
    influential_papers: List[Dict]
    related_topics: List[Dict]
    timeline_analysis: Dict[str, Any]

class AIModelManager:
    """AI 模型管理器"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """加载模型"""
        if self.is_loaded:
            return True
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("transformers 库未安装，无法加载AI模型")
            return False
            
        try:
            logger.info(f"开始加载 AI 模型: {self.model_path}")
            start_time = time.time()
            
            # 加载 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.is_loaded = True
            
            load_time = time.time() - start_time
            logger.info(f"AI 模型加载完成，耗时: {load_time:.2f}秒")
            return True
            
        except Exception as e:
            logger.error(f"加载 AI 模型失败: {str(e)}")
            logger.error(traceback.format_exc())
            self.is_loaded = False
            return False
    
    def generate_response(self, prompt: str, max_length: int = 1024, temperature: float = 0.7) -> str:
        """生成 AI 回复"""
        if not self.is_loaded:
            return "AI 模型未加载，请稍后再试。"
        
        try:
            # 构建简单的对话格式
            text = f"""你是一个专业的研究助手，专门分析学术主题和研究趋势。请基于提供的数据给出准确、有用的回答。

用户提问：{prompt}

请给出专业、准确、有帮助的回答："""
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码回复
            response = outputs[0][inputs["input_ids"].shape[1]:]
            response_text = self.tokenizer.decode(response, skip_special_tokens=True)
            
            return response_text.strip()
            
        except Exception as e:
            logger.error(f"AI 生成失败: {str(e)}")
            logger.error(traceback.format_exc())
            return "抱歉，生成回复时出现错误。"
    
    def analyze_topic(self, topic_id: int, data_cache: Dict) -> Optional[TopicAnalysisResult]:
        """分析特定主题"""
        if not self.is_loaded:
            if not self.load_model():
                return None
        
        # 获取主题信息
        if data_cache["topics_info_df"] is None:
            return None
            
        topic_info = data_cache["topics_info_df"][data_cache["topics_info_df"]["Topic"] == topic_id]
        if topic_info.empty:
            return None
            
        topic_name = topic_info.iloc[0]["Name"]
        
        # 获取该主题的论文
        papers_df = data_cache["papers_df"]
        if papers_df is None:
            return None
            
        topic_papers = papers_df[papers_df["new_topic"] == topic_id]
        
        # 准备分析提示词
        prompt = self._build_topic_analysis_prompt(topic_id, topic_name, topic_papers, data_cache)
        
        try:
            # 生成分析
            response = self.generate_response(prompt, max_length=1500)
            
            # 解析响应
            return self._parse_topic_analysis(response, topic_id, topic_name, topic_papers)
            
        except Exception as e:
            logger.error(f"主题分析失败: {str(e)}")
            return None
    
    def _build_topic_analysis_prompt(self, topic_id: int, topic_name: str, 
                                   topic_papers: pd.DataFrame, data_cache: Dict) -> str:
        """构建主题分析提示词"""
        
        # 获取相关统计信息
        paper_count = len(topic_papers) if not topic_papers.empty else 0
        avg_citations = topic_papers["n_citation"].mean() if not topic_papers.empty and "n_citation" in topic_papers.columns else 0
        
        # 获取关键词
        keywords_list = []
        if data_cache["topic_keywords_df"] is not None:
            keywords_df = data_cache["topic_keywords_df"]
            if "Topic" in keywords_df.columns and "Word" in keywords_df.columns:
                topic_keywords = keywords_df[keywords_df["Topic"] == topic_id]
                if not topic_keywords.empty:
                    keywords_list = topic_keywords["Word"].head(10).tolist()
        
        # 构建提示词
        prompt = f"""请分析学术主题 "{topic_name}" (ID: {topic_id})。

基础信息:
- 论文数量: {paper_count}
- 平均引用数: {avg_citations:.1f}

关键词: {', '.join(keywords_list[:5]) if keywords_list else "暂无关键词"}

请从以下几个方面进行分析:
1. 主题概括: 简要描述这个研究主题的核心内容
2. 关键概念: 列出3-5个最核心的研究概念或方向
3. 研究趋势: 分析近年来的研究热点和变化趋势
4. 影响力分析: 基于论文引用情况，评估该主题的影响力
5. 相关主题: 建议2-3个密切相关的研究方向
6. 未来展望: 提出该领域可能的发展方向

请使用结构化的方式回答，每个部分清晰明了。"""
        
        return prompt
    
    def _parse_topic_analysis(self, response: str, topic_id: int, topic_name: str, 
                            topic_papers: pd.DataFrame) -> TopicAnalysisResult:
        """解析 AI 响应"""
        # 找出高影响力论文
        top_papers = []
        if not topic_papers.empty and "n_citation" in topic_papers.columns and "title" in topic_papers.columns:
            try:
                top_papers = topic_papers.nlargest(5, "n_citation")[["title", "n_citation", "year"]].to_dict(orient="records")
            except Exception as e:
                logger.error(f"提取高影响力论文失败: {e}")
        
        # 简单解析响应内容
        sections = self._extract_sections(response)
        
        return TopicAnalysisResult(
            topic_id=topic_id,
            topic_name=topic_name,
            summary=sections.get("主题概括", response[:200] + "..." if len(response) > 200 else response),
            key_concepts=self._extract_list(sections.get("关键概念", "")),
            research_trends=self._extract_list(sections.get("研究趋势", "")),
            influential_papers=top_papers,
            related_topics=[],
            timeline_analysis={}
        )
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """从文本中提取各个部分"""
        sections = {}
        current_section = None
        lines = text.split('\n')
        
        for line in lines:
            # 查找章节标题（以数字或中文标识）
            section_match = re.match(r'^(?:\d+[\.、]?\s*)?([^:：]+)[:：]\s*(.*)', line.strip())
            if section_match:
                current_section = section_match.group(1).strip()
                sections[current_section] = section_match.group(2).strip()
            elif current_section and line.strip():
                sections[current_section] += '\n' + line.strip()
        
        return sections
    
    def _extract_list(self, text: str) -> List[str]:
        """从文本中提取列表项"""
        items = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # 匹配列表项（以数字、符号开头）
            if re.match(r'^[•\-*\d\.、]\s*(.+)', line):
                item = re.sub(r'^[•\-*\d\.、]\s*', '', line)
                if item:
                    items.append(item)
        return items if items else [text] if text else []

# 初始化 AI 模型管理器
ai_manager = AIModelManager(MODEL_PATH, DEVICE)

# -------------------- 数据加载函数 -------------------- #
def load_data():
    """加载所有必要的数据"""
    global _data_cache
    
    try:
        # 加载论文数据
        if _data_cache["papers_df"] is None:
            logger.info("加载论文数据...")
            if DATA_FILE.exists():
                _data_cache["papers_df"] = pd.read_json(DATA_FILE)
                logger.info(f"论文数据加载完成，共 {len(_data_cache['papers_df'])} 篇论文")
            else:
                logger.error(f"论文数据文件不存在: {DATA_FILE}")
        
        # 加载主题信息
        if _data_cache["topics_info_df"] is None:
            logger.info("加载主题信息...")
            info_path = DATA_DIR / "new_topics_info.csv"
            if info_path.exists():
                _data_cache["topics_info_df"] = pd.read_csv(info_path)
                logger.info(f"主题信息加载完成，共 {len(_data_cache['topics_info_df'])} 个主题")
            else:
                logger.error(f"主题信息文件不存在: {info_path}")
        
        # 加载关键词
        if _data_cache["topic_keywords_df"] is None:
            logger.info("加载主题关键词...")
            kw_path = DATA_DIR / "new_topic_keywords_weight.csv"
            if kw_path.exists():
                _data_cache["topic_keywords_df"] = pd.read_csv(kw_path)
                logger.info(f"关键词数据加载完成，共 {len(_data_cache['topic_keywords_df'])} 行")
            else:
                logger.warning(f"关键词数据文件不存在: {kw_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# -------------------- AI 对话接口 -------------------- #
@app.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    """AI 对话接口"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "请求体为空"}), 400
        
        message = data.get('message', '').strip()
        topic_id = data.get('topic')
        
        if not message:
            return jsonify({"error": "消息内容不能为空"}), 400
        
        # 加载数据
        if not load_data():
            return jsonify({"error": "数据加载失败"}), 500
        
        # 确保 AI 模型已加载
        if not ai_manager.is_loaded:
            if not ai_manager.load_model():
                return jsonify({"error": "AI模型加载失败"}), 500
        
        # 构建上下文感知的提示词
        context_prompt = build_chat_prompt(message, topic_id)
        
        # 生成回复
        start_time = time.time()
        response = ai_manager.generate_response(context_prompt)
        response_time = time.time() - start_time
        
        logger.info(f"AI 响应生成完成，耗时: {response_time:.2f}秒")
        
        # 构建返回数据
        result = {
            "response": response,
            "topic_id": topic_id,
            "timestamp": datetime.now().isoformat(),
            "response_time": f"{response_time:.2f}秒"
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"AI 对话接口错误: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"内部错误: {str(e)}"}), 500

def build_chat_prompt(user_message: str, topic_id: Optional[int] = None) -> str:
    """构建聊天提示词"""
    
    context_parts = []
    
    # 如果有主题ID，添加主题上下文
    if topic_id is not None and _data_cache["topics_info_df"] is not None:
        if topic_id in _data_cache["topics_info_df"]["Topic"].values:
            topic_info = _data_cache["topics_info_df"][_data_cache["topics_info_df"]["Topic"] == topic_id]
            if not topic_info.empty:
                topic_name = topic_info.iloc[0]["Name"]
                context_parts.append(f"当前用户正在研究主题：{topic_name} (ID: {topic_id})")
                
                # ✅ 修复：添加具体的论文数据
                if _data_cache["papers_df"] is not None and "new_topic" in _data_cache["papers_df"].columns:
                    topic_papers = _data_cache["papers_df"][_data_cache["papers_df"]["new_topic"] == topic_id]
                    paper_count = len(topic_papers)
                    
                    # 添加2021年之后的论文信息
                    if "year" in topic_papers.columns:
                        recent_papers = topic_papers[topic_papers["year"] >= 2021]
                        recent_papers = recent_papers.sort_values("n_citation", ascending=False).head(10)
                        
                        if len(recent_papers) > 0:
                            context_parts.append("\n相关论文数据（2021-2025年）：")
                            for idx, paper in recent_papers.iterrows():
                                title = paper.get("title", "无标题")
                                year = paper.get("year", "未知")
                                citations = paper.get("n_citation", 0)
                                authors = paper.get("authors", [])
                                author_names = [a.get("name", "") for a in authors if isinstance(authors, list)]
                                
                                context_parts.append(
                                    f"{idx+1}. {title} (年份: {year}, 引用: {citations})"
                                )
    
    # 添加明确的时间约束到提示词
    time_constraint = "\n重要：请主要参考2021-2025年的数据，如果没有相关数据请明确说明。"
    
    if context_parts:
        context = "\n".join(context_parts) + time_constraint
        prompt = f"""{context}

用户提问：{user_message}

请基于以上提供的数据（特别是2021-2025年的论文信息），给出专业、准确、有帮助的回答。"""
    else:
        prompt = f"""{time_constraint}

用户提问：{user_message}

请基于学术研究相关知识，给出专业、准确、有帮助的回答。"""
    
    return prompt

# -------------------- 主题分析接口 -------------------- #
@app.route('/api/ai/analyze_topic', methods=['GET'])
def analyze_topic():
    """深度分析特定主题"""
    try:
        topic_id = request.args.get('topic_id', type=int)
        if topic_id is None:
            return jsonify({"error": "缺少 topic_id 参数"}), 400
        
        # 加载数据
        if not load_data():
            return jsonify({"error": "数据加载失败"}), 500
        
        # 确保 AI 模型已加载
        if not ai_manager.is_loaded:
            if not ai_manager.load_model():
                return jsonify({"error": "AI模型加载失败"}), 500
        
        # 检查主题是否存在
        if _data_cache["topics_info_df"] is None:
            return jsonify({"error": "主题信息未加载"}), 500
        
        # 修复：正确检查主题是否存在
        if topic_id not in _data_cache["topics_info_df"]["Topic"].values:
            return jsonify({"error": f"主题 {topic_id} 不存在"}), 404
        
        # 分析主题
        logger.info(f"开始分析主题 {topic_id}")
        start_time = time.time()
        analysis = ai_manager.analyze_topic(topic_id, _data_cache)
        analysis_time = time.time() - start_time
        
        if analysis is None:
            return jsonify({"error": "主题分析失败"}), 500
        
        logger.info(f"主题分析完成，耗时: {analysis_time:.2f}秒")
        
        # 转换为字典格式
        result = {
            "topic_id": analysis.topic_id,
            "topic_name": analysis.topic_name,
            "summary": analysis.summary,
            "key_concepts": analysis.key_concepts,
            "research_trends": analysis.research_trends,
            "influential_papers": analysis.influential_papers,
            "related_topics": analysis.related_topics,
            "timeline_analysis": analysis.timeline_analysis,
            "analysis_time": f"{analysis_time:.2f}秒",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"主题分析接口错误: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"内部错误: {str(e)}"}), 500

# -------------------- 快速问答接口 -------------------- #
@app.route('/api/ai/quick_questions', methods=['GET'])
def get_quick_questions():
    """获取针对特定主题的快速问题建议"""
    try:
        topic_id = request.args.get('topic_id', type=int)
        if topic_id is None:
            return jsonify({"error": "缺少 topic_id 参数"}), 400
        
        # 检查主题是否存在
        if _data_cache["topics_info_df"] is None:
            if not load_data():
                return jsonify({"error": "数据加载失败"}), 500
        
        # 修复：正确检查主题是否存在
        if topic_id not in _data_cache["topics_info_df"]["Topic"].values:
            return jsonify({"error": f"主题 {topic_id} 不存在"}), 404
        
        # 获取主题名称
        topic_info = _data_cache["topics_info_df"][_data_cache["topics_info_df"]["Topic"] == topic_id]
        topic_name = topic_info.iloc[0]["Name"] if not topic_info.empty else f"主题 {topic_id}"
        
        # 预定义的快速问题
        quick_questions = [
            f"分析主题'{topic_name}'的研究趋势",
            f"解释{ topic_name}的主要研究方向",
            f"推荐与{ topic_name}相关的高影响力论文",
            f"{ topic_name}与其他主题的关系如何？",
            f"近年来{ topic_name}领域有哪些重要突破？",
            f"如何开始研究{ topic_name}这个领域？",
            f"{ topic_name}领域的经典理论有哪些？",
            f"预测{ topic_name}未来五年的发展方向"
        ]
        
        return jsonify({
            "topic_id": topic_id,
            "topic_name": topic_name,
            "questions": quick_questions
        })
        
    except Exception as e:
        logger.error(f"快速问题接口错误: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"内部错误: {str(e)}"}), 500

# -------------------- AI 模型状态接口 -------------------- #
@app.route('/api/ai/status', methods=['GET'])
def ai_status():
    """获取 AI 模型状态"""
    status = {
        "model_loaded": ai_manager.is_loaded,
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "transformers_available": TRANSFORMERS_AVAILABLE,
        "data_loaded": _data_cache["papers_df"] is not None and _data_cache["topics_info_df"] is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    if ai_manager.is_loaded:
        # 添加模型信息
        status["model_info"] = {
            "vocab_size": len(ai_manager.tokenizer) if ai_manager.tokenizer else 0,
        }
        if DEVICE == "cuda" and torch.cuda.is_available():
            status["model_info"]["device_memory"] = torch.cuda.memory_allocated() / 1024**3
    
    # 这里直接使用 jsonify
    return jsonify(status)

# ============================================================================== #
#  以下是原有接口（包含修复）
# ============================================================================== #

# 预加载推理后主题关键词表（只跑一次）
_NEW_KW_PATH = DATA_DIR / "new_topic_keywords_weight.csv"
_topic_texts = []          # 每个元素对应一个主题，格式 "word1 word2 word3 ..."
_topic_ids = []            # 与 _topic_texts 同序的主题 id
_vectorizer = TfidfVectorizer()
_tfidf_matrix = None       #  shape = (n_topics, n_features)
NOISE_TOPICS = {-1, 0, 20}   # 凡是觉得"太宽泛"的主题号写这里

def _load_topic_corpus():
    global _topic_texts, _topic_ids, _tfidf_matrix
    if _tfidf_matrix is not None:
        return
    if not _NEW_KW_PATH.exists():
        return
    df = pd.read_csv(_NEW_KW_PATH)
    # ===== 不再过滤，保留全部主题 =====
    topn = 30
    grouped = (df.sort_values(["Topic", "Weight"], ascending=False)
                 .groupby("Topic").head(topn).groupby("Topic")["Word"]
                 .apply(lambda x: " ".join(x)))
    _topic_ids = grouped.index.astype(int).tolist()
    _topic_texts = grouped.values.tolist()
    _tfidf_matrix = _vectorizer.fit_transform(_topic_texts)
    print('[DEBUG] 加载完成：主题数={}, shape={}'.format(
          len(_topic_ids), _tfidf_matrix.shape))

_load_topic_corpus()

@app.route("/api/recommend_topic")
def recommend_topic():
    """?q=用户文本 → 返回最相似主题 id 与名称"""
    try:
        q = request.args.get("q", "").strip()
        if not q:
            return jsonify({"error": "empty query"}), 400
        if _tfidf_matrix is None:
            return jsonify({"error": "corpus not loaded"}), 500
        q_vec = _vectorizer.transform([q])
        sim = cosine_similarity(q_vec, _tfidf_matrix)[0]   # 1×n_topics
        for idx, tid in enumerate(_topic_ids):
            if tid in NOISE_TOPICS:
                sim[idx] = -float('inf')
        best_idx = int(sim.argmax())
        best_id = int(_topic_ids[best_idx])
        best_score = float(sim[best_idx])
        if best_score == -float('inf'):
            return jsonify({"topic_id": None, "topic_name": None, "score": 0.0})
        # 顺便把名称带回去
        info_df = pd.read_csv(DATA_DIR / "new_topics_info.csv")
        name_map = {int(r.Topic): str(r.Name) for _, r in info_df.iterrows()}
        return jsonify({
            "topic_id": best_id,
            "topic_name": name_map.get(best_id, f"主题 {best_id}"),
            "score": best_score
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================== #
#  新增接口：主题相似度映射
# ============================================================================== #

@app.route('/api/topic_similarity_old')
def get_topic_similarity_old():
    try:
        if not os.path.exists(SIM_FILE_OLD):
            return jsonify({"error": f"Not found: {SIM_FILE_OLD}"}), 404
        with open(SIM_FILE_OLD, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/topic_similarity_new')
def get_topic_similarity_new():
    try:
        if not os.path.exists(SIM_FILE_NEW):
            return jsonify({"error": f"Not found: {SIM_FILE_NEW}"}), 404
        with open(SIM_FILE_NEW, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500  

# ============================================================================== #
#  以下全部为原有接口，逻辑未动，仅统一缩进与类型注解
# ============================================================================== #
@app.route("/api/topic_tree_old")
def get_topic_tree_old():
    """获取原始主题的层次树图数据"""
    try:
        kw_path = DATA_DIR / "topic_keywords_weight.csv"
        kw_df = pd.read_csv(kw_path)
        matrix = kw_df.pivot(index="Topic", columns="Word", values="Weight").fillna(0)
        topic_ids = matrix.index.map(int).tolist()

        dist_matrix = 1 - cosine_similarity(matrix)
        Z = linkage(dist_matrix, method="ward")

        info_df = pd.read_csv(DATA_DIR / "topics_info.csv")
        nodes_info = {
            int(row["Topic"]): {
                "id": int(row["Topic"]),
                "name": str(row["Name"]),
                "count": int(row["Count"]),
            }
            for _, row in info_df.iterrows()
        }

        tree = hierarchy_to_tree(Z, topic_ids, nodes_info)
        return jsonify(
            {
                "linkage": Z.tolist(),
                "topic_ids": topic_ids,
                "nodes": list(nodes_info.values()),
                "tree": tree,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Topic tree old error: {str(e)}"}), 500


@app.route("/api/topic_tree_new")
def get_topic_tree_new():
    """获取新主题的层次树图数据"""
    try:
        kw_new_path = DATA_DIR / "new_topic_keywords_weight.csv"
        kw_old_path = DATA_DIR / "topic_keywords_weight.csv"

        if kw_new_path.exists():
            kw_df = pd.read_csv(kw_new_path)
        elif kw_old_path.exists():
            kw_df = pd.read_csv(kw_old_path)
        else:
            return jsonify({"error": "No keyword weight file found"}), 404

        matrix = kw_df.pivot(index="Topic", columns="Word", values="Weight").fillna(0)
        all_topic_ids = (
            pd.read_csv(DATA_DIR / "new_topics_info.csv")["Topic"]
            .astype(int)
            .tolist()
        )
        matrix = matrix.reindex(all_topic_ids).fillna(0)

        Z = linkage(1 - cosine_similarity(matrix), method="ward") if len(matrix) > 1 else []

        info_df = pd.read_csv(DATA_DIR / "new_topics_info.csv")
        nodes_info = {
            int(row["Topic"]): {
                "id": int(row["Topic"]),
                "name": str(row["Name"]),
                "count": int(row.get("Count", 0)),
                "is_new": int(row["Topic"]) >= 20,
            }
            for _, row in info_df.iterrows()
        }

        tree = hierarchy_to_tree(Z, all_topic_ids, nodes_info)
        return jsonify(
            {
                "linkage": Z.tolist() if len(Z) else [],
                "topic_ids": all_topic_ids,
                "nodes": list(nodes_info.values()),
                "tree": tree,
                "debug": {
                    "matrix_shape": matrix.shape,
                    "topic_count": len(all_topic_ids),
                    "nodes_count": len(nodes_info),
                    "linkage_shape": Z.shape if len(Z) else [0, 0],
                },
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Topic tree new error: {str(e)}"}), 500


def hierarchy_to_tree(Z, ids, nodes_info):
    """将 scipy 层次聚类结果转为前端树结构"""
    if len(Z) == 0:
        if len(ids) == 1:
            tid = ids[0]
            return {
                "id": tid,
                "name": nodes_info.get(tid, {}).get("name", f"主题 {tid}"),
                "count": nodes_info.get(tid, {}).get("count", 0),
            }
        return None

    scipy_tree = scipy_to_tree(Z, rd=False)

    def build_node(node):
        if node.is_leaf():
            tid = ids[node.id]
            return {
                "id": tid,
                "name": nodes_info.get(tid, {}).get("name", f"主题 {tid}"),
                "count": nodes_info.get(tid, {}).get("count", 0),
                "distance": float(node.dist) if hasattr(node, "dist") else 0,
            }
        return {
            "id": f"cluster_{node.id}",
            "distance": float(node.dist) if hasattr(node, "dist") else 0,
            "children": [build_node(node.get_left()), build_node(node.get_right())],
        }

    return build_node(scipy_tree)


# -------------------- 其他旧接口 -------------------- #
@app.route("/api/topic_hierarchy")
def get_hierarchy():
    try:
        kw_df = pd.read_csv(DATA_DIR / "topic_keywords_weight.csv")
        matrix = kw_df.pivot(index="Topic", columns="Word", values="Weight").fillna(0)
        Z = linkage(1 - cosine_similarity(matrix), method="ward")
        info_df = pd.read_csv(DATA_DIR / "topics_info.csv")
        names = {int(row["Topic"]): str(row["Name"]) for _, row in info_df.iterrows()}
        return jsonify(
            {
                "linkage": Z.tolist(),
                "topic_ids": matrix.index.map(int).tolist(),
                "names": names,
            }
        )
    except Exception as e:
        return jsonify({"error": f"Hierarchy error: {str(e)}"}), 500


# -------------------- 噪声重分配：紧凑流转视图 -------------------- #
@app.route("/api/viz/module2_v4_logic")
def get_v4_logic_data():
    """
    为 InferenceFlow.vue 提供噪声重分配数据
    """
    try:
        file_path = DATA_DIR / "final_docs_data.json"
        info_path = DATA_DIR / "new_topics_info.csv"
        old_info_path = DATA_DIR / "topics_info.csv"

        if not file_path.exists():
            return jsonify({"error": "Data file not found"}), 404

        df = pd.read_json(file_path)
        if "topic" not in df.columns or "new_topic" not in df.columns:
            return jsonify({"error": "Invalid data格式"}), 500

        # 统计噪声分配
        noise_df = df[df["topic"] == -1]
        total_noise = len(noise_df)
        stats = noise_df["new_topic"].value_counts().to_dict()
        stats = {int(k): int(v) for k, v in stats.items()}

        # 主题名称映射
        name_map = {}
        for p in (old_info_path, info_path):
            if p.exists():
                tmp = pd.read_csv(p)
                name_map.update(
                    {int(row.Topic): str(row.Name) for _, row in tmp.iterrows()}
                )

        # 左侧节点：-1 + 存量 0-19
        left_nodes = [{"id": -1, "name": name_map.get(-1, "原始噪声池"), "value": total_noise}]
        left_nodes.extend(
            {"id": i, "name": name_map.get(i, f"主题 {i}"), "value": stats.get(i, 0)}
            for i in range(20)
        )

        # 右侧新主题
        right_nodes = [
            {"id": tid, "name": name_map.get(tid, f"主题 {tid}"), "value": cnt}
            for tid, cnt in stats.items()
            if tid >= 20
        ]

        return jsonify(
            {
                "noise_total": total_noise,
                "left_nodes": left_nodes,
                "right_nodes": right_nodes,
                "topic_names": name_map,
            }
        )
    except Exception as e:
        logging.exception("v4_logic error")
        return jsonify({"error": str(e)}), 500

# -------------------- 占位 / 其他接口 -------------------- #
@app.route("/api/topic_minus_one_new")
def get_minus_one_new():
    return jsonify({"topic": -1, "value": 0.05, "name": "Noise/New"}), 200


@app.route("/api/sankey_data")
def get_sankey_data():
    try:
        with open("sankey_data.json", encoding="utf-8") as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({"nodes": [], "links": []})

import networkx as nx
import numpy as np
# ========== 合著网络：按主题分别 Top50 + 全主题饼图 ========== #
@app.route('/api/collaboration')
def collaboration():
    """合著网络接口 - 恢复原逻辑"""
    topic_filter = request.args.get('topic', type=int, default=None)
    
    try:
        # 检查数据文件
        if not DATA_FILE.exists():
            logger.error(f"数据文件不存在: {DATA_FILE}")
            return jsonify({'nodes': [], 'links': [], 'error': '数据文件不存在'}), 200
        
        # 读取数据
        with DATA_FILE.open(encoding='utf-8') as f:
            all_papers = json.load(f)
        
        # 调试信息：记录论文总数
        logger.info(f"总论文数: {len(all_papers)}")
        
        # 1. 构建「主题 → 论文列表」映射
        topic2papers = defaultdict(list)
        for p in all_papers:
            # 获取主题ID，默认-1
            new_topic = p.get('new_topic', -1)
            # 确保主题ID是整数
            try:
                new_topic = int(new_topic)
            except (ValueError, TypeError):
                new_topic = -1
            topic2papers[new_topic].append(p)
        
        # 调试信息：记录主题分布
        logger.info(f"主题数量: {len(topic2papers)}")
        for topic_id, papers in list(topic2papers.items())[:10]:  # 只显示前10个主题
            logger.info(f"主题 {topic_id}: {len(papers)} 篇论文")
        
        # 2. 确定要处理的主题集合
        if topic_filter is None:
            # 默认：处理所有主题，每个主题取Top 50
            target_topics = list(topic2papers.keys())
            logger.info(f"处理所有主题，共 {len(target_topics)} 个主题")
        else:
            # 只处理指定主题
            target_topics = [topic_filter]
            logger.info(f"只处理主题: {topic_filter}")
        
        # 3. 每个主题分别「引用量降序 → Top50」合并成统一池子
        top_pool = []
        for tid in target_topics:
            papers = topic2papers.get(tid, [])
            if not papers:
                continue
                
            # 按引用量排序，取前50
            # 确保每篇论文都有引用数字段
            sorted_papers = sorted(
                papers, 
                key=lambda p: p.get('n_citation', 0), 
                reverse=True
            )
            top50 = sorted_papers[:50]
            top_pool.extend(top50)
            logger.info(f"主题 {tid}: 选取 {len(top50)} 篇论文 (总计 {len(papers)} 篇)")
        
        logger.info(f"最终论文池大小: {len(top_pool)} 篇论文")
        
        # 4. 收集池子里所有作者 ID
        author_ids_in_pool = set()
        for p in top_pool:
            authors = p.get('authors', [])
            for a in authors:
                author_id = a.get('id')
                if author_id:
                    author_ids_in_pool.add(author_id)
        
        logger.info(f"作者池大小: {len(author_ids_in_pool)} 位作者")
        
        # 5. 全库累加：统计这些作者的全部主题和引用
        author_map = {}
        for p in all_papers:
            authors = p.get('authors', [])
            for a in authors:
                author_id = a.get('id')
                if not author_id or author_id not in author_ids_in_pool:
                    continue
                    
                if author_id not in author_map:
                    author_map[author_id] = {
                        'id': author_id,
                        'name': a.get('name', '未知作者'),
                        'org': a.get('org', '未知机构'),
                        'n_citation': 0,
                        'article_count': 0,
                        'topics': defaultdict(int),
                        'articles': []
                    }
                
                nd = author_map[author_id]
                nd['article_count'] += 1
                nd['n_citation'] += p.get('n_citation', 0)
                
                # 获取论文主题
                paper_topic = p.get('new_topic', -1)
                try:
                    paper_topic = int(paper_topic)
                except (ValueError, TypeError):
                    paper_topic = -1
                
                nd['topics'][paper_topic] += 1
                
                # 只把「池子内论文」放进 articles
                if p in top_pool:
                    nd['articles'].append({
                        'id': p.get('id', ''),
                        'title': p.get('title', '无标题'),
                        'topic': paper_topic
                    })
        
        # 6. 边：只在 top_pool 论文里全连接作者
        edges = defaultdict(int)
        for p in top_pool:
            authors = p.get('authors', [])
            aids = [a.get('id') for a in authors if a.get('id')]
            
            # 全连接同一论文的所有作者
            for i in range(len(aids)):
                for j in range(i + 1, len(aids)):
                    key = tuple(sorted([aids[i], aids[j]]))
                    edges[key] += 1
        
        # 7. 构建节点列表
        nodes = []
        for aid, nd in author_map.items():
            node = {
                'id': aid,
                'name': nd['name'],
                'org': nd['org'],
                'n_citation': nd['n_citation'],
                'article_count': nd['article_count'],
                'topics': [{'topic': int(k), 'count': v} for k, v in nd['topics'].items()],
                'articles': nd['articles']
            }
            nodes.append(node)
        
        # 构建边列表
            # ========== 线条瘦身开始 ==========
        w_threshold = 1 if topic_filter is None else 1
        edges = {k: w for k, w in edges.items() if w >= w_threshold}

        G = nx.Graph()
        G.add_edges_from(edges)
        lcc = max(nx.connected_components(G), key=len)
        edges = {k: w for k, w in edges.items() if k[0] in lcc and k[1] in lcc}

        links = [{'source': s, 'target': t, 'weight': w}
             for (s, t), w in edges.items()]

        keep_ids = {s for s, _ in edges} | {t for _, t in edges}
        nodes  = [n for n in nodes if n['id'] in keep_ids]
        # ========== 线条瘦身结束 ==========

    # 原有保险截断
        nodes = nodes[:1000]
        keep_ids = {n['id'] for n in nodes}
        links = [lk for lk in links
             if lk['source'] in keep_ids and lk['target'] in keep_ids][:5000]

        logger.info(f'瘦身后 => 节点 {len(nodes)}  边 {len(links)}')
        
        # 8. 截断（防止超大网络）
        # original_node_count = len(nodes)
        # original_link_count = len(links)
        
        # nodes = nodes[:1000]
        # keep_ids = {n['id'] for n in nodes}
        # links = [l for l in links if l['source'] in keep_ids and l['target'] in keep_ids][:5000]
        
        # logger.info(f"最终网络: {len(nodes)} 节点 (原 {original_node_count}), {len(links)} 边 (原 {original_link_count})")
        
        # 9. 检查是否有数据
        if len(nodes) == 0:
            logger.warning("网络节点数为0，可能数据有问题")
            # 添加调试信息
            debug_info = {
                'total_papers': len(all_papers),
                'top_pool_size': len(top_pool),
                'author_pool_size': len(author_ids_in_pool),
                'author_map_size': len(author_map),
                'target_topics': target_topics,
                'topic_paper_counts': {tid: len(papers) for tid, papers in topic2papers.items() if tid in target_topics}
            }
            logger.info(f"调试信息: {debug_info}")
        
        return jsonify({
            'nodes': nodes,
            'links': links,
            'debug': {
                'total_papers': len(all_papers),
                'top_pool_size': len(top_pool),
                'nodes_count': len(nodes),
                'links_count': len(links)
            }
        })
    
    except FileNotFoundError:
        logger.error(f"文件未找到: {DATA_FILE}")
        return jsonify({'nodes': [], 'links': [], 'error': '数据文件未找到'}), 200
    except Exception as e:
        logger.error(f"合著网络接口错误: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'nodes': [], 'links': [], 'error': str(e)}), 500

# ========== 主题名称（防止 404）========== #
@app.route('/api/topic-names')
def topic_names():
    try:
        info = Path(__file__).with_name('data') / 'new_topics_info.csv'
        if not info.exists():
            return jsonify([])
        df = pd.read_csv(info)
        return jsonify([{'id': int(row.Topic), 'name': str(row.Name)} for _, row in df.iterrows()])
    except Exception as e:
        logger.error(f"获取主题名称错误: {e}")
        return jsonify([])

@app.route('/api/papers')
def papers():
    try:
        # 修复：直接使用 DATA_FILE
        if not DATA_FILE.exists():
            return jsonify([]), 200
            
        df = pd.read_json(DATA_FILE)
        print('① 刚读进来列名：', df.columns.tolist())          # 第一步
        print('② 有没有 n_citation：', 'n_citation' in df.columns)

        cols = ['id', 'new_topic', 'title', 'n_citation',
                'abstract', 'year', 'doi', 'venue','authors']
        df = df.reindex(columns=cols)
        print('③ reindex 后列名：', df.columns.tolist())        # 第三步
        print('④ reindex 后 n_citation 存在：', 'n_citation' in df.columns)

        # 2. 补默认值
        df['n_citation'] = df['n_citation'].fillna(0).astype(int)
        df['doi']        = df['doi'].fillna('')
        df['abstract']   = df['abstract'].fillna('')
        df['year']       = df['year'].fillna('')
        df['venue']       = df['venue'].fillna('')
        df['authors']    = df['authors'].fillna({})

        # 3. 主题过滤
        topic = request.args.get('topic', type=int)
        if topic is not None:
            df = df[df['new_topic'] == topic]

        # 4. 引用量倒序
        df = df.sort_values('n_citation', ascending=False)

        return jsonify(df.to_dict(orient='records'))
    except FileNotFoundError:
        return jsonify([]), 200
    except Exception as e:
        logger.error(f"获取论文列表错误: {e}")
        return jsonify([]), 200

# -------------------- 模型初始化 -------------------- #
# @app.before_first_request
# def initialize():
#     """应用启动时的初始化"""
#     logger.info("正在初始化应用...")
    
#     # 加载数据
#     load_data()
    
#     # 异步加载 AI 模型（避免阻塞启动）
#     def load_model_async():
#         try:
#             if TRANSFORMERS_AVAILABLE:
#                 ai_manager.load_model()
#                 logger.info("AI 模型初始化完成")
#             else:
#                 logger.warning("transformers 库未安装，跳过AI模型加载")
#         except Exception as e:
#             logger.error(f"AI 模型初始化失败: {str(e)}")
    
#     import threading
#     thread = threading.Thread(target=load_model_async)
#     thread.daemon = True
#     thread.start()
    
#     logger.info("应用初始化完成")

# 在 app.py 中的合适位置（比如其他数据加载函数附近）添加以下代码

@app.route('/api/wordcloud/all', methods=['GET'])
def get_all_topics_wordcloud():
    """获取所有主题的词云数据（最重要的不同关键词）"""
    try:
        # 加载关键词数据
        keywords_path = DATA_DIR / "new_topic_keywords_weight.csv"
        if not keywords_path.exists():
            return jsonify({"error": "关键词文件不存在"}), 404
        
        df = pd.read_csv(keywords_path)
        
        # 过滤掉噪声主题
        df = df[~df["Topic"].isin([-1, 0, 20])]
        
        # 按主题和权重分组，获取每个主题的前5个关键词
        grouped = df.sort_values("Weight", ascending=False).groupby("Topic")
        
        all_words = []
        seen_words = set()  # 用于去重，确保关键词不重复
        
        # 遍历每个主题，取权重最高的关键词
        for topic_id, group in grouped:
            # 获取这个主题的前5个关键词
            top_keywords = group.head(5).to_dict('records')
            
            for kw in top_keywords:
                word = kw.get("Word", "").strip()
                weight = kw.get("Weight", 0)
                
                # 如果这个词还没出现过，就添加到结果中
                if word and word not in seen_words:
                    seen_words.add(word)
                    all_words.append({
                        "text": word,
                        "size": float(weight) * 100,  # 放大以便显示
                        "topic": int(topic_id),
                        "weight": float(weight)
                    })
        
        # 按权重排序并限制数量
        all_words.sort(key=lambda x: x["weight"], reverse=True)
        all_words = all_words[:50]  # 最多显示50个
        
        # 去掉weight字段，只保留前端需要的字段
        result = []
        for word in all_words:
            result.append({
                "text": word["text"],
                "size": word["size"],
                "topic": word["topic"]
            })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"获取全主题词云数据失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/wordcloud/topic/<int:topic_id>', methods=['GET'])
def get_topic_wordcloud(topic_id):
    """获取特定主题的词云数据"""
    try:
        # 加载关键词数据
        keywords_path = DATA_DIR / "new_topic_keywords_weight.csv"
        if not keywords_path.exists():
            return jsonify({"error": "关键词文件不存在"}), 404
        
        df = pd.read_csv(keywords_path)
        
        # 筛选指定主题的关键词
        topic_df = df[df["Topic"] == topic_id]
        
        if topic_df.empty:
            # 如果在新主题中找不到，尝试在旧主题中找
            old_keywords_path = DATA_DIR / "topic_keywords_weight.csv"
            if old_keywords_path.exists():
                old_df = pd.read_csv(old_keywords_path)
                topic_df = old_df[old_df["Topic"] == topic_id]
        
        if topic_df.empty:
            return jsonify([]), 200
        
        # 按权重排序，取前30个关键词
        topic_df = topic_df.sort_values("Weight", ascending=False).head(30)
        
        # 构建词云数据
        words = []
        max_weight = topic_df["Weight"].max()
        min_weight = topic_df["Weight"].min()
        
        for _, row in topic_df.iterrows():
            word = row.get("Word", "").strip()
            weight = float(row.get("Weight", 0))
            
            if word:
                # 归一化权重到合适的显示大小（10-40）
                if max_weight > min_weight:
                    size = 10 + (weight - min_weight) / (max_weight - min_weight) * 30
                else:
                    size = 25
                
                words.append({
                    "text": word,
                    "size": size,
                    "topic": int(topic_id),
                    "weight": weight
                })
        
        return jsonify(words)
        
    except Exception as e:
        logger.error(f"获取主题词云数据失败: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/wordcloud/topic_names', methods=['GET'])
def get_topic_names_for_wordcloud():
    """获取主题名称映射，供词云组件使用"""
    try:
        # 尝试加载新主题信息
        info_path = DATA_DIR / "new_topics_info.csv"
        if not info_path.exists():
            # 回退到旧主题信息
            info_path = DATA_DIR / "topics_info.csv"
        
        if not info_path.exists():
            return jsonify({}), 200
        
        df = pd.read_csv(info_path)
        name_map = {}
        
        for _, row in df.iterrows():
            topic_id = int(row.get("Topic", 0))
            topic_name = str(row.get("Name", f"主题 {topic_id}"))
            name_map[topic_id] = topic_name
        
        return jsonify(name_map)
        
    except Exception as e:
        logger.error(f"获取主题名称失败: {str(e)}")
        return jsonify({}), 200


# ====================== 修复后的机构统计接口 ====================== #
@app.route('/api/org_topic_stats', methods=['GET'])
def get_org_topic_stats():
    """获取机构的主题统计信息 - 修复版，支持机构名称模糊匹配"""
    # 在函数开头导入jsonify
    from flask import jsonify
    
    try:
        # 加载数据
        if not load_data():
            logger.error("数据加载失败")
            return jsonify({"error": "数据加载失败", "debug": "load_data() returned False"}), 500
        
        papers_df = _data_cache["papers_df"]
        if papers_df is None or len(papers_df) == 0:
            logger.error("论文数据未加载或为空")
            return jsonify({
                "error": "论文数据未加载", 
                "debug": f"papers_df is None: {papers_df is None}",
                "data_file_exists": DATA_FILE.exists()
            }), 500
        
        logger.info(f"读取到论文数据: {len(papers_df)} 行")
        logger.info(f"数据列名: {papers_df.columns.tolist()}")
        
        # 检查是否有authors列
        if 'authors' not in papers_df.columns:
            logger.error("数据中没有authors列")
            return jsonify({
                "error": "数据格式不正确，缺少authors列",
                "actual_columns": papers_df.columns.tolist()
            }), 500
        
        # 获取参数
        org_limit = request.args.get('limit', type=int, default=10)
        
        # 1. 按机构统计总引用量 - 使用机构名称规范化
        org_citations = {}  # 统一机构名 -> 总引用量
        org_papers = {}     # 统一机构名 -> 论文列表
        org_original_names = {}  # 统一机构名 -> 原始机构名（用于显示）
        processed_count = 0
        author_count = 0
        
        for idx, paper in papers_df.iterrows():
            authors = paper.get('authors')
            
            # 检查authors数据格式
            if authors is None or (isinstance(authors, float) and math.isnan(authors)):
                continue
                
            # 如果是字符串，尝试解析
            if isinstance(authors, str):
                try:
                    # 处理JSON格式的字符串
                    if authors.startswith('[') and authors.endswith(']'):
                        import ast
                        authors = ast.literal_eval(authors)
                    else:
                        import json
                        authors = json.loads(authors)
                except Exception as e:
                    logger.warning(f"解析authors失败 (row {idx}): {str(e)[:50]}")
                    continue
            # 如果是列表但不是字典列表，跳过
            elif isinstance(authors, list):
                # 检查列表中的元素是否为字典
                if len(authors) > 0 and not isinstance(authors[0], dict):
                    continue
            else:
                continue
            
            if not authors or len(authors) == 0:
                continue
                
            # 处理引用量
            try:
                citations = int(float(paper.get('n_citation', 0)))
            except:
                citations = 0
            
            paper_added_to_org = False
            for author in authors:
                if not isinstance(author, dict):
                    continue
                    
                org = author.get('org')
                # 修复：检查org是否为None
                if org is None:
                    continue
                    
                # 安全地处理org值
                if isinstance(org, str):
                    org_name = org.strip()
                else:
                    # 如果是数字或其他类型，转换为字符串
                    try:
                        org_name = str(org).strip()
                    except:
                        continue
                
                if not org_name or org_name.lower() in ['', 'none', 'null', 'nan', 'unknown']:
                    continue
                
                # ========== 新增：规范化机构名称 ==========
                # 统一机构名称，使用标准化名称进行统计
                unified_org_name = unify_org_name(org_name)
                
                # 如果没有统一结果，使用规范化后的名称
                if not unified_org_name:
                    unified_org_name = normalize_org_name(org_name)
                
                # 如果还是没有，使用原始名称
                if not unified_org_name:
                    unified_org_name = org_name
                
                # 统计机构引用量
                if unified_org_name not in org_citations:
                    org_citations[unified_org_name] = 0
                    org_papers[unified_org_name] = []
                    # 保存原始机构名称，用于显示
                    org_original_names[unified_org_name] = org_name
                
                # 每篇论文只添加一次到机构
                if not paper_added_to_org:
                    # 获取论文主题
                    try:
                        topic = int(float(paper.get('new_topic', -1)))
                    except:
                        topic = -1
                    
                    org_papers[unified_org_name].append({
                        'paper_id': paper.get('id', f'paper_{idx}'),
                        'topic': topic,
                        'citations': citations,
                        'year': paper.get('year', ''),
                        'title': paper.get('title', '无标题'),
                        'original_org_name': org_name  # 保存原始机构名称
                    })
                    paper_added_to_org = True
                
                org_citations[unified_org_name] += citations
                author_count += 1
            
            processed_count += 1
        
        logger.info(f"数据处理完成:")
        logger.info(f"  成功处理论文: {processed_count}")
        logger.info(f"  处理作者: {author_count}")
        logger.info(f"  发现机构数: {len(org_citations)}")
        
        # 如果没有机构数据，返回友好的空结果
        if not org_citations:
            return jsonify({
                "orgs": [],
                "palette": {},
                "stats": {},
                "message": "未找到有效的机构数据，请检查数据格式",
                "debug": {
                    "total_papers": len(papers_df),
                    "processed_papers": processed_count,
                    "total_authors": author_count
                }
            }), 200
        
        # 2. 按引用量排序，取前N个机构
        top_orgs = sorted(org_citations.items(), key=lambda x: x[1], reverse=True)[:org_limit]
        
        # 3. 为每个机构统计主题分布
        result = {
            'orgs': [],
            'palette': {},
            'stats': {},
            'debug': {
                'total_orgs_found': len(org_citations),
                'top_orgs_count': len(top_orgs),
                'processed_papers': processed_count
            }
        }
        
        # 获取所有主题ID
        all_topics = set()
        for org_data in org_papers.values():
            for paper in org_data:
                all_topics.add(paper['topic'])
        
        # 创建主题颜色映射
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]
        
        topics_list = sorted(list(all_topics))
        for idx, topic_id in enumerate(topics_list):
            result['palette'][topic_id] = colors[idx % len(colors)]
        
        # 统计每个机构的主题分布
        for unified_org_name, total_citations in top_orgs:
            papers_list = org_papers.get(unified_org_name, [])
            
            if not papers_list:
                continue
            
            # 使用原始机构名称进行显示
            display_name = org_original_names.get(unified_org_name, unified_org_name)
            
            # 统计主题频次和引用量
            topic_stats = {}
            for paper in papers_list:
                topic = paper['topic']
                if topic not in topic_stats:
                    topic_stats[topic] = {
                        'count': 0,
                        'citations': 0,
                        'papers': []
                    }
                
                topic_stats[topic]['count'] += 1
                topic_stats[topic]['citations'] += paper['citations']
                topic_stats[topic]['papers'].append({
                    'id': paper['paper_id'],
                    'title': paper['title'],
                    'year': paper['year'],
                    'citations': paper['citations']
                })
            
            # 转换为数组格式
            topic_distribution = []
            for topic_id, stats in topic_stats.items():
                topic_distribution.append({
                    'topic_id': topic_id,
                    'topic_name': f"主题 {topic_id}",
                    'paper_count': stats['count'],
                    'citation_count': stats['citations'],
                    'percentage': (stats['count'] / len(papers_list)) * 100 if papers_list else 0,
                    'papers': stats['papers'][:5]
                })
            
            # 按论文数量排序
            topic_distribution.sort(key=lambda x: x['paper_count'], reverse=True)
            
            result['orgs'].append({
                'name': display_name,  # 使用原始机构名称显示
                'unified_name': unified_org_name,  # 保存统一名称用于匹配
                'total_citations': total_citations,
                'paper_count': len(papers_list),
                'topic_distribution': topic_distribution
            })
            
            # 为每个机构保存详细统计
            result['stats'][display_name] = {
                'topics': topic_distribution,
                'total_papers': len(papers_list)
            }
        
        # 按总引用量排序机构
        result['orgs'].sort(key=lambda x: x['total_citations'], reverse=True)
        
        logger.info(f"API返回 {len(result['orgs'])} 个机构的数据")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"机构主题统计接口错误: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"内部错误: {str(e)}"}), 500


@app.route('/api/topic_org_stats', methods=['GET'])
def get_topic_org_stats():
    """获取特定主题的机构统计 - 修复版，支持机构名称模糊匹配"""
    try:
        topic_id = request.args.get('topic_id', type=int)
        limit = request.args.get('limit', type=int, default=10)
        
        if topic_id is None:
            return jsonify({"error": "需要 topic_id 参数"}), 400
        
        # 加载数据
        if not load_data():
            return jsonify({"error": "数据加载失败"}), 500
        
        papers_df = _data_cache["papers_df"]
        if papers_df is None:
            return jsonify({"error": "论文数据未加载"}), 500
        
        logger.info(f"查询主题 {topic_id} 的机构统计")
        
        # 检查列名
        logger.info(f"数据列名: {papers_df.columns.tolist()}")
        
        # 确保有 new_topic 列
        if 'new_topic' not in papers_df.columns:
            return jsonify({"error": "数据中没有 new_topic 列"}), 500
        
        # 筛选该主题的论文
        try:
            # 转换为数字类型
            papers_df['new_topic'] = pd.to_numeric(papers_df['new_topic'], errors='coerce')
            topic_papers = papers_df[papers_df['new_topic'] == topic_id]
        except Exception as e:
            logger.error(f"筛选主题论文失败: {str(e)}")
            return jsonify({"error": f"筛选数据失败: {str(e)}"}), 500
        
        logger.info(f"找到主题 {topic_id} 的论文: {len(topic_papers)} 篇")
        
        if len(topic_papers) == 0:
            return jsonify({
                "topic_id": topic_id,
                "topic_name": f"主题 {topic_id}",
                "orgs": [],
                "total_orgs": 0,
                "total_papers": 0,
                "message": "该主题没有论文数据"
            }), 200
        
        # 统计机构 - 使用机构名称规范化
        org_stats = {}  # 统一机构名 -> 统计信息
        org_original_names = {}  # 统一机构名 -> 原始机构名
        
        for idx, paper in topic_papers.iterrows():
            authors = paper.get('authors')
            
            # 处理authors数据
            if isinstance(authors, float) and math.isnan(authors):
                continue
                
            if isinstance(authors, str) and authors:
                try:
                    if authors.startswith('[') and authors.endswith(']'):
                        import ast
                        authors = ast.literal_eval(authors)
                except:
                    continue
            elif not isinstance(authors, list):
                continue
            
            if not authors or len(authors) == 0:
                continue
            
            # 统计每个机构
            paper_unified_orgs = set()
            paper_original_orgs = {}  # 统一名 -> 原始名
            
            for author in authors:
                if not isinstance(author, dict):
                    continue
                    
                org = author.get('org')
                if org is None:
                    continue
                
                # 安全处理机构名称
                if isinstance(org, str):
                    org_name = org.strip()
                else:
                    try:
                        org_name = str(org).strip()
                    except:
                        continue
                
                if not org_name or org_name.lower() in ['', 'none', 'null', 'nan']:
                    continue
                
                # ========== 新增：规范化机构名称 ==========
                # 统一机构名称
                unified_org_name = unify_org_name(org_name)
                
                # 如果没有统一结果，使用规范化后的名称
                if not unified_org_name:
                    unified_org_name = normalize_org_name(org_name)
                
                # 如果还是没有，使用原始名称
                if not unified_org_name:
                    unified_org_name = org_name
                
                paper_unified_orgs.add(unified_org_name)
                paper_original_orgs[unified_org_name] = org_name
            
            # 为每个机构添加统计
            try:
                citations = int(float(paper.get('n_citation', 0)))
            except:
                citations = 0
            
            for unified_org_name in paper_unified_orgs:
                if unified_org_name not in org_stats:
                    org_stats[unified_org_name] = {
                        'papers': [],
                        'total_citations': 0,
                        'paper_count': 0,
                        'original_name': paper_original_orgs.get(unified_org_name, unified_org_name)
                    }
                    org_original_names[unified_org_name] = paper_original_orgs.get(unified_org_name, unified_org_name)
                
                org_stats[unified_org_name]['papers'].append({
                    'id': paper.get('id', f'paper_{idx}'),
                    'title': paper.get('title', '无标题'),
                    'citations': citations,
                    'year': paper.get('year', ''),
                    'original_org_name': paper_original_orgs.get(unified_org_name, unified_org_name)
                })
                org_stats[unified_org_name]['total_citations'] += citations
                org_stats[unified_org_name]['paper_count'] += 1
        
        logger.info(f"主题 {topic_id} 涉及 {len(org_stats)} 个机构")
        
        # 转换为前端需要的格式
        orgs_list = []
        for unified_org_name, stats in org_stats.items():
            # 使用原始机构名称进行显示
            display_name = org_original_names.get(unified_org_name, unified_org_name)
            
            orgs_list.append({
                'name': display_name,
                'unified_name': unified_org_name,
                'paper_count': stats['paper_count'],
                'total_citations': stats['total_citations'],
                'topic_distribution': [{
                    'topic_id': topic_id,
                    'topic_name': f"主题 {topic_id}",
                    'paper_count': stats['paper_count'],
                    'citation_count': stats['total_citations'],
                    'percentage': 100.0
                }],
                'original_name': stats['original_name']
            })
        
        # 按引用量排序（降序）
        orgs_list.sort(key=lambda x: x['total_citations'], reverse=True)
        
        # 只取前limit个
        if limit > 0:
            orgs_list = orgs_list[:limit]
        
        # 获取主题名称
        topic_name = f"主题 {topic_id}"
        if _data_cache["topics_info_df"] is not None:
            topic_info = _data_cache["topics_info_df"][_data_cache["topics_info_df"]["Topic"] == topic_id]
            if not topic_info.empty:
                topic_name = topic_info.iloc[0]["Name"]
        
        return jsonify({
            'topic_id': topic_id,
            'topic_name': topic_name,
            'orgs': orgs_list,
            'total_orgs': len(org_stats),
            'total_papers': len(topic_papers),
            'debug_info': {
                'query_topic': topic_id,
                'found_papers': len(topic_papers),
                'found_orgs': len(org_stats),
                'returned_orgs': len(orgs_list)
            }
        })
        
    except Exception as e:
        logger.error(f"主题机构统计接口错误: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"内部错误: {str(e)}"}), 500


@app.route('/api/org_detailed_stats', methods=['GET'])
def get_org_detailed_stats():
    """获取特定机构的详细统计信息 - 修复版，支持机构名称模糊匹配"""
    # 在函数开头导入jsonify
    from flask import jsonify
    
    try:
        org_name = request.args.get('name', '')
        if not org_name:
            return jsonify({"error": "机构名称不能为空"}), 400
        
        # 加载数据
        if not load_data():
            return jsonify({"error": "数据加载失败"}), 500
        
        papers_df = _data_cache["papers_df"]
        if papers_df is None:
            return jsonify({"error": "论文数据未加载"}), 500
        
        logger.info(f"开始查询机构: {org_name}")
        
        # 统一查询的机构名称
        unified_query_name = unify_org_name(org_name)
        if not unified_query_name:
            unified_query_name = normalize_org_name(org_name)
        
        logger.info(f"查询统一机构名: {unified_query_name}")
        
        # 1. 收集该机构的所有论文
        org_papers = []
        for idx, paper in papers_df.iterrows():
            authors = paper.get('authors')
            
            # 处理authors数据
            if authors is None or (isinstance(authors, float) and math.isnan(authors)):
                continue
                
            # 如果是字符串，尝试解析
            if isinstance(authors, str):
                try:
                    if authors.startswith('[') and authors.endswith(']'):
                        import ast
                        authors = ast.literal_eval(authors)
                    else:
                        import json
                        authors = json.loads(authors)
                except:
                    continue
            
            if not isinstance(authors, list) or len(authors) == 0:
                continue
                
            # 检查是否有作者属于该机构（使用统一名称匹配）
            found = False
            for author in authors:
                if not isinstance(author, dict):
                    continue
                    
                author_org = author.get('org')
                if author_org is None:
                    continue
                
                # 处理作者机构名称
                if isinstance(author_org, str):
                    author_org_name = author_org.strip()
                else:
                    try:
                        author_org_name = str(author_org).strip()
                    except:
                        continue
                
                if not author_org_name:
                    continue
                
                # 统一作者机构名称
                unified_author_org = unify_org_name(author_org_name)
                if not unified_author_org:
                    unified_author_org = normalize_org_name(author_org_name)
                
                # 检查是否匹配（使用统一名称）
                if (unified_author_org == unified_query_name or 
                    (unified_query_name and unified_query_name in unified_author_org) or
                    (unified_author_org and unified_author_org in unified_query_name)):
                    found = True
                    break
            
            if found:
                try:
                    topic = int(float(paper.get('new_topic', -1)))
                except:
                    topic = -1
                    
                try:
                    citations = int(float(paper.get('n_citation', 0)))
                except:
                    citations = 0
                
                org_papers.append({
                    'id': paper.get('id', f'paper_{idx}'),
                    'title': paper.get('title', '无标题'),
                    'topic': topic,
                    'citations': citations,
                    'year': paper.get('year', ''),
                    'abstract': paper.get('abstract', ''),
                    'venue': paper.get('venue', ''),
                    'authors': authors,
                    'doi': paper.get('doi', '')
                })
        
        logger.info(f"找到 {len(org_papers)} 篇属于机构 {org_name} 的论文")
        
        if not org_papers:
            return jsonify({
                "org_name": org_name,
                "error": "未找到该机构的论文",
                "suggestions": "请检查机构名称是否正确，或尝试其他机构"
            }), 200
        
        # 2. 按主题分组统计
        topic_stats = {}
        for paper in org_papers:
            topic = paper['topic']
            if topic not in topic_stats:
                topic_stats[topic] = {
                    'papers': [],
                    'total_citations': 0,
                    'paper_count': 0,
                    'years': set(),
                    'authors': set()
                }
            
            topic_stats[topic]['papers'].append(paper)
            topic_stats[topic]['total_citations'] += paper['citations']
            topic_stats[topic]['paper_count'] += 1
            topic_stats[topic]['years'].add(paper['year'])
            
            # 收集作者
            for author in paper['authors']:
                if isinstance(author, dict):
                    author_id = author.get('id', author.get('name', ''))
                    if author_id:
                        topic_stats[topic]['authors'].add(author_id)
        
        # 3. 转换为前端需要的格式
        topics_data = []
        for topic_id, stats in topic_stats.items():
            # 计算时间范围
            years = sorted([y for y in stats['years'] if y and str(y).isdigit()])
            time_range = f"{min(years)}-{max(years)}" if years else "未知"
            
            # 获取主题名称
            topic_name = f"主题 {topic_id}"
            if _data_cache["topics_info_df"] is not None:
                topic_info = _data_cache["topics_info_df"][_data_cache["topics_info_df"]["Topic"] == topic_id]
                if not topic_info.empty:
                    topic_name = topic_info.iloc[0]["Name"]
            
            # 获取高影响力论文
            high_impact_papers = sorted(
                stats['papers'], 
                key=lambda x: x['citations'], 
                reverse=True
            )[:3]
            
            topics_data.append({
                'topic_id': topic_id,
                'topic_name': topic_name,
                'paper_count': stats['paper_count'],
                'total_citations': stats['total_citations'],
                'avg_citations': stats['total_citations'] / stats['paper_count'] if stats['paper_count'] > 0 else 0,
                'time_range': time_range,
                'author_count': len(stats['authors']),
                'high_impact_papers': [
                    {
                        'title': p['title'],
                        'citations': p['citations'],
                        'year': p['year'],
                        'authors': [a.get('name', '') for a in p['authors'][:3] if isinstance(a, dict)]
                    }
                    for p in high_impact_papers
                ]
            })
        
        # 按论文数量排序
        topics_data.sort(key=lambda x: x['paper_count'], reverse=True)
        
        # 4. 统计年度趋势
        year_stats = {}
        for paper in org_papers:
            year = paper['year']
            if year:
                if year not in year_stats:
                    year_stats[year] = {'count': 0, 'citations': 0}
                year_stats[year]['count'] += 1
                year_stats[year]['citations'] += paper['citations']
        
        # 转换为排序后的数组
        year_trend = [
            {
                'year': year,
                'paper_count': stats['count'],
                'citation_count': stats['citations']
            }
            for year, stats in sorted(year_stats.items())
        ]
        
        # 5. 统计主要作者
        author_stats = {}
        for paper in org_papers:
            for author in paper['authors']:
                if isinstance(author, dict):
                    author_id = author.get('id', author.get('name', ''))
                    author_name = author.get('name', '未知作者')
                    
                    if not author_id:
                        continue
                        
                    if author_id not in author_stats:
                        author_stats[author_id] = {
                            'name': author_name,
                            'papers': 0,
                            'citations': 0
                        }
                    
                    author_stats[author_id]['papers'] += 1
                    author_stats[author_id]['citations'] += paper['citations']
        
        # 转换为数组并排序
        top_authors = sorted(
            [
                {
                    'id': author_id,
                    'name': stats['name'],
                    'paper_count': stats['papers'],
                    'citation_count': stats['citations'],
                    'avg_citations': stats['citations'] / stats['papers'] if stats['papers'] > 0 else 0
                }
                for author_id, stats in author_stats.items()
            ],
            key=lambda x: x['paper_count'],
            reverse=True
        )[:10]
        
        result = {
            'org_name': org_name,
            'summary': {
                'total_papers': len(org_papers),
                'total_citations': sum(p['citations'] for p in org_papers),
                'topic_count': len(topic_stats),
                'author_count': len(author_stats),
                'years_covered': f"{min(years) if years else '未知'}-{max(years) if years else '未知'}"
            },
            'topics': topics_data,
            'year_trend': year_trend,
            'top_authors': top_authors,
            'top_papers': sorted(org_papers, key=lambda x: x['citations'], reverse=True)[:10]
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"机构详细统计接口错误: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"内部错误: {str(e)}"}), 500


# 添加一个简单的测试接口，帮助调试
@app.route('/api/debug/authors_sample', methods=['GET'])
def debug_authors_sample():
    """调试接口：查看authors数据的格式"""
    from flask import jsonify
    
    try:
        if not load_data():
            return jsonify({"error": "数据加载失败"}), 500
        
        papers_df = _data_cache["papers_df"]
        if papers_df is None or len(papers_df) == 0:
            return jsonify({"error": "论文数据未加载"}), 500
        
        # 获取几行示例数据
        samples = []
        for i in range(min(5, len(papers_df))):
            paper = papers_df.iloc[i]
            authors = paper.get('authors')
            
            sample = {
                'index': i,
                'authors_type': type(authors).__name__,
                'authors_value': str(authors)[:200] if authors else None,
                'has_org_column': 'org' in paper
            }
            
            # 尝试解析
            if isinstance(authors, str):
                try:
                    if authors.startswith('[') and authors.endswith(']'):
                        import ast
                        parsed = ast.literal_eval(authors)
                        sample['parsed_type'] = type(parsed).__name__
                        sample['parsed_length'] = len(parsed) if isinstance(parsed, list) else 'N/A'
                        if isinstance(parsed, list) and len(parsed) > 0:
                            sample['first_author'] = str(parsed[0])[:100]
                except Exception as e:
                    sample['parse_error'] = str(e)
            
            samples.append(sample)
        
        return jsonify({
            'total_papers': len(papers_df),
            'columns': papers_df.columns.tolist(),
            'samples': samples
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====================== 新增：机构筛选接口 ====================== #
@app.route('/api/org_filtered_collaboration', methods=['GET'])
def get_org_filtered_collaboration():
    """获取特定机构的合著网络数据"""
    try:
        org_name = request.args.get('org', '').strip()
        if not org_name:
            return jsonify({"error": "机构名称不能为空"}), 400
        
        print(f"查询机构合著网络，机构: {org_name}")
        
        # 加载数据
        if not load_data():
            return jsonify({"error": "数据加载失败"}), 500
        
        papers_df = _data_cache["papers_df"]
        if papers_df is None:
            return jsonify({"error": "论文数据未加载"}), 500
        
        # 统一查询的机构名称
        unified_query_name = unify_org_name(org_name)
        if not unified_query_name:
            unified_query_name = normalize_org_name(org_name)
        
        print(f"统一机构名: {unified_query_name}")
        
        # 1. 收集该机构的所有论文
        org_papers = []
        paper_ids = set()
        
        for idx, paper in papers_df.iterrows():
            authors = paper.get('authors')
            
            # 处理authors数据
            if authors is None or (isinstance(authors, float) and math.isnan(authors)):
                continue
                
            # 如果是字符串，尝试解析
            if isinstance(authors, str):
                try:
                    if authors.startswith('[') and authors.endswith(']'):
                        import ast
                        authors = ast.literal_eval(authors)
                    else:
                        import json
                        authors = json.loads(authors)
                except:
                    continue
            
            if not isinstance(authors, list) or len(authors) == 0:
                continue
                
            # 检查是否有作者属于该机构（使用统一名称匹配）
            found = False
            for author in authors:
                if not isinstance(author, dict):
                    continue
                    
                author_org = author.get('org')
                if author_org is None:
                    continue
                
                # 处理作者机构名称
                if isinstance(author_org, str):
                    author_org_name = author_org.strip()
                else:
                    try:
                        author_org_name = str(author_org).strip()
                    except:
                        continue
                
                if not author_org_name:
                    continue
                
                # 统一作者机构名称
                unified_author_org = unify_org_name(author_org_name)
                if not unified_author_org:
                    unified_author_org = normalize_org_name(author_org_name)
                
                # 检查是否匹配（使用统一名称）
                if (unified_author_org == unified_query_name or 
                    (unified_query_name and unified_query_name in unified_author_org) or
                    (unified_author_org and unified_author_org in unified_query_name)):
                    found = True
                    break
            
            if found:
                try:
                    topic = int(float(paper.get('new_topic', -1)))
                except:
                    topic = -1
                    
                try:
                    citations = int(float(paper.get('n_citation', 0)))
                except:
                    citations = 0
                
                paper_data = {
                    'id': paper.get('id', f'paper_{idx}'),
                    'title': paper.get('title', '无标题'),
                    'topic': topic,
                    'n_citation': citations,
                    'citations': citations,
                    'year': paper.get('year', ''),
                    'abstract': paper.get('abstract', ''),
                    'venue': paper.get('venue', ''),
                    'authors': authors,
                    'doi': paper.get('doi', ''),
                    'new_topic': topic
                }
                org_papers.append(paper_data)
                paper_ids.add(paper.get('id', f'paper_{idx}'))
        
        print(f"找到 {len(org_papers)} 篇属于机构 {org_name} 的论文")
        
        if len(org_papers) == 0:
            return jsonify({
                'nodes': [],
                'links': [],
                'debug': {
                    'org_name': org_name,
                    'unified_name': unified_query_name,
                    'paper_count': 0,
                    'message': '该机构没有论文数据'
                }
            })
        
        # 2. 收集所有相关作者
        author_map = {}
        author_ids_in_org = set()
        
        for paper in org_papers:
            authors = paper.get('authors', [])
            for author in authors:
                if not isinstance(author, dict):
                    continue
                    
                author_id = author.get('id')
                if not author_id:
                    continue
                    
                author_ids_in_org.add(author_id)
                
                if author_id not in author_map:
                    author_map[author_id] = {
                        'id': author_id,
                        'name': author.get('name', '未知作者'),
                        'org': author.get('org', '未知机构'),
                        'n_citation': 0,
                        'article_count': 0,
                        'topics': defaultdict(int),
                        'articles': []
                    }
                
                nd = author_map[author_id]
                nd['article_count'] += 1
                nd['n_citation'] += paper.get('n_citation', 0)
                
                # 获取论文主题
                paper_topic = paper.get('topic', -1)
                nd['topics'][paper_topic] += 1
                
                # 把论文信息添加到作者的文章列表
                nd['articles'].append({
                    'id': paper.get('id', ''),
                    'title': paper.get('title', '无标题'),
                    'topic': paper_topic
                })
        
        print(f"找到 {len(author_map)} 位作者")
        
        # 3. 构建合著边（只在机构内论文中）
        edges = defaultdict(int)
        for paper in org_papers:
            authors = paper.get('authors', [])
            aids = [a.get('id') for a in authors if a.get('id')]
            
            # 全连接同一论文的所有作者
            for i in range(len(aids)):
                for j in range(i + 1, len(aids)):
                    key = tuple(sorted([aids[i], aids[j]]))
                    edges[key] += 1
        
        print(f"生成 {len(edges)} 条合著边")
        
        # 4. 构建节点列表
        nodes = []
        for aid, nd in author_map.items():
            # 将topics字典转换为数组
            topics_array = []
            if nd['topics']:
                for topic_id, count in nd['topics'].items():
                    topics_array.append({
                        'topic': int(topic_id),
                        'count': count
                    })
            
            node = {
                'id': aid,
                'name': nd['name'],
                'org': nd['org'],
                'n_citation': nd['n_citation'],
                'citation': nd['n_citation'],
                'citations': nd['n_citation'],
                'article_count': nd['article_count'],
                'topics': topics_array,
                'articles': nd['articles']
            }
            nodes.append(node)
        
        # 5. 构建边列表
        links = []
        for (source_id, target_id), weight in edges.items():
            # 确保节点存在
            if source_id in author_map and target_id in author_map:
                links.append({
                    'source': source_id,
                    'target': target_id,
                    'weight': weight
                })
        
        print(f"最终网络: {len(nodes)} 节点, {len(links)} 边")
        
        # 6. 应用瘦身逻辑（如果网络太大）
        if len(nodes) > 50:
            print("网络较大，应用瘦身逻辑...")
            # 按论文数量排序，保留前50个作者
            nodes.sort(key=lambda x: x['article_count'], reverse=True)
            nodes = nodes[:50]
            
            # 更新作者ID集合
            keep_ids = {n['id'] for n in nodes}
            
            # 过滤边，只保留保留节点之间的边
            links = [lk for lk in links 
                    if lk['source'] in keep_ids and lk['target'] in keep_ids]
            
            print(f"瘦身后: {len(nodes)} 节点, {len(links)} 边")
        
        return jsonify({
            'nodes': nodes,
            'links': links,
            'debug': {
                'org_name': org_name,
                'unified_name': unified_query_name,
                'paper_count': len(org_papers),
                'author_count': len(nodes),
                'link_count': len(links),
                'query_time': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        print(f"机构合著网络接口错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'nodes': [], 'links': [], 'error': str(e)}), 500
    

@app.route('/api/org_filtered_papers', methods=['GET'])
def get_org_filtered_papers():
    """获取特定机构的论文列表"""
    try:
        org_name = request.args.get('org', '').strip()
        if not org_name:
            return jsonify({"error": "机构名称不能为空"}), 400
        
        print(f"查询机构论文列表，机构: {org_name}")
        
        # 加载数据
        if not load_data():
            return jsonify({"error": "数据加载失败"}), 500
        
        papers_df = _data_cache["papers_df"]
        if papers_df is None:
            return jsonify({"error": "论文数据未加载"}), 500
        
        # 统一查询的机构名称
        unified_query_name = unify_org_name(org_name)
        if not unified_query_name:
            unified_query_name = normalize_org_name(org_name)
        
        # 1. 收集该机构的所有论文
        org_papers = []
        
        for idx, paper in papers_df.iterrows():
            authors = paper.get('authors')
            
            # 处理authors数据
            if authors is None or (isinstance(authors, float) and math.isnan(authors)):
                continue
                
            # 如果是字符串，尝试解析
            if isinstance(authors, str):
                try:
                    if authors.startswith('[') and authors.endswith(']'):
                        import ast
                        authors = ast.literal_eval(authors)
                    else:
                        import json
                        authors = json.loads(authors)
                except:
                    continue
            
            if not isinstance(authors, list) or len(authors) == 0:
                continue
                
            # 检查是否有作者属于该机构（使用统一名称匹配）
            found = False
            for author in authors:
                if not isinstance(author, dict):
                    continue
                    
                author_org = author.get('org')
                if author_org is None:
                    continue
                
                # 处理作者机构名称
                if isinstance(author_org, str):
                    author_org_name = author_org.strip()
                else:
                    try:
                        author_org_name = str(author_org).strip()
                    except:
                        continue
                
                if not author_org_name:
                    continue
                
                # 统一作者机构名称
                unified_author_org = unify_org_name(author_org_name)
                if not unified_author_org:
                    unified_author_org = normalize_org_name(author_org_name)
                
                # 检查是否匹配（使用统一名称）
                if (unified_author_org == unified_query_name or 
                    (unified_query_name and unified_query_name in unified_author_org) or
                    (unified_author_org and unified_author_org in unified_query_name)):
                    found = True
                    break
            
            if found:
                try:
                    topic = int(float(paper.get('new_topic', -1)))
                except:
                    topic = -1
                    
                try:
                    citations = int(float(paper.get('n_citation', 0)))
                except:
                    citations = 0
                
                # 构建论文数据
                paper_data = {
                    'id': paper.get('id', f'paper_{idx}'),
                    'title': paper.get('title', '无标题'),
                    'new_topic': topic,
                    'n_citation': citations,
                    'abstract': paper.get('abstract', ''),
                    'venue': paper.get('venue', ''),
                    'year': paper.get('year', ''),
                    'doi': paper.get('doi', ''),
                    'authors': authors
                }
                org_papers.append(paper_data)
        
        print(f"找到 {len(org_papers)} 篇论文")
        
        # 2. 按引用量排序
        org_papers.sort(key=lambda x: x.get('n_citation', 0), reverse=True)
        
        # 3. 限制返回数量（避免数据量过大）
        if len(org_papers) > 1000:
            org_papers = org_papers[:1000]
            print(f"限制返回数量为 1000 篇")
        
        return jsonify(org_papers)
        
    except Exception as e:
        print(f"机构论文列表接口错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify([]), 500

# -------------------- 主入口 -------------------- #
if __name__ == "__main__":
    # 启动时初始化
    print(f"数据目录: {DATA_DIR}")
    print(f"数据文件: {DATA_FILE}")
    print(f"数据文件存在: {DATA_FILE.exists()}")
    
    # initialize()
    app.run(port=5000, debug=True, threaded=True)