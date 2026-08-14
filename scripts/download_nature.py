#!/usr/bin/env python3
"""
PubMed Nature 系列杂志文章下载器
通过参数指定杂志名称、下载年份、杂志编号，使用 E-utilities API 检索并下载 PDF。
示例：--journal-name "Nature communications" --year 2024 --journal-id 41467
"""

import os
import re
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
from urllib.parse import quote, urlencode
from xml.etree import ElementTree as ET

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gear.env import getenv

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUTS_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f'nature_download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PubMedNatureDownloader:
    """使用PubMed E-utilities API的Nature文章下载器"""
    
    # E-utilities API 端点
    EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
    EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"
    
    # Nature PDF基础URL
    NATURE_PDF_BASE = "https://www.nature.com/articles"
    
    def __init__(self, email: str = "your-email@example.com", 
                 tool: str = "PubMedNatureDownloader",
                 api_key: str = None):
        """
        初始化下载器
        
        Args:
            email: 您的邮箱（NCBI要求，用于问题联系）
            tool: 工具名称
            api_key: NCBI API密钥（可选，但有密钥可提高速率限制）
        """
        self.email = email
        self.tool = tool
        self.api_key = api_key
        
        self.session = requests.Session()
        
        # 设置合理的请求头
        self.session.headers.update({
            'User-Agent': f'{tool}/1.0 ({email})',
            'Accept': 'application/xml',
            'Accept-Encoding': 'gzip, deflate',
        })
        
        # API请求参数
        self.delay = 0.34  # 约3次/秒，符合NCBI建议
        self.timeout = 30
        self.max_retries = 3
        
        # DOI缓存文件
        self.dois_cache_file = "dois_cache.txt"
        
        # 统计信息
        self.total_found = 0
        self.dois_fetched = 0
        self.pdfs_downloaded = 0
        
    def search_articles(self, year: int = 2023, journal_name: str = None, retmax: int = 100000) -> List[str]:
        """
        搜索指定杂志指定年份文章，返回PMID列表
        
        Args:
            year: 年份
            journal_name: 杂志名称（PubMed 期刊名），若为 None 则使用 run() 时设置的 _journal_name
            retmax: 最大返回结果数
            
        Returns:
            PMID列表
        """
        name = journal_name if journal_name is not None else getattr(self, '_journal_name', 'Nature Biomedical Engineering')
        logger.info(f"正在搜索 {year} 年 {name} 文章...")
        
        # 构建PubMed搜索查询
        search_term = f'("{name}"[Journal]) AND ("{year}/01/01"[Date - Publication] : "{year}/12/31"[Date - Publication])'
        
        params = {
            'db': 'pubmed',
            'term': search_term,
            'retmax': retmax,
            'retmode': 'json',
            'usehistory': 'y',  # 使用历史记录，提高效率
            'tool': self.tool,
            'email': self.email,
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        try:
            # 执行搜索
            time.sleep(self.delay)
            response = self.session.get(self.ESEARCH_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            # 获取结果
            self.total_found = int(data.get('esearchresult', {}).get('count', 0))
            pmid_list = data.get('esearchresult', {}).get('idlist', [])
            
            logger.info(f"找到 {self.total_found} 篇文章")
            logger.info(f"获取到 {len(pmid_list)} 个PMID")
            
            # 保存查询历史，用于后续批量获取
            self.query_key = data.get('esearchresult', {}).get('querykey', '1')
            self.webenv = data.get('esearchresult', {}).get('webenv', '')
            
            return pmid_list
            
        except Exception as e:
            logger.error(f"搜索文章失败: {e}")
            if hasattr(response, 'text'):
                logger.error(f"响应内容: {response.text[:500]}")
            return []
    
    def fetch_dois_from_pmids(self, pmid_list: List[str], batch_size: int = 200) -> List[str]:
        """
        通过PMID批量获取DOI
        
        Args:
            pmid_list: PMID列表
            batch_size: 每批处理的数量
            
        Returns:
            DOI列表
        """
        if not pmid_list:
            return []
        
        logger.info(f"开始获取 {len(pmid_list)} 篇文章的DOI...")
        
        all_dois = []
        total_batches = (len(pmid_list) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(pmid_list))
            batch_pmids = pmid_list[start_idx:end_idx]
            
            logger.info(f"处理批次 {batch_idx+1}/{total_batches}: PMID {start_idx+1}-{end_idx}")
            
            try:
                # 构建批处理参数
                params = {
                    'db': 'pubmed',
                    'id': ','.join(batch_pmids),
                    'retmode': 'xml',
                    'rettype': 'abstract',
                    'tool': self.tool,
                    'email': self.email,
                }
                
                if self.api_key:
                    params['api_key'] = self.api_key
                
                # 使用历史记录（如果可用）
                if hasattr(self, 'webenv') and self.webenv:
                    params['WebEnv'] = self.webenv
                    params['query_key'] = self.query_key
                
                # 获取文章详情
                time.sleep(self.delay)
                response = self.session.get(self.EFETCH_URL, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                # 解析XML，提取DOI
                batch_dois = self._extract_dois_from_xml(response.text)
                
                all_dois.extend(batch_dois)
                
                logger.info(f"批次 {batch_idx+1} 获取到 {len(batch_dois)} 个DOI")
                
                # 批次间延迟
                if batch_idx < total_batches - 1:
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"批次 {batch_idx+1} 获取DOI失败: {e}")
                continue
        
        self.dois_fetched = len(all_dois)
        logger.info(f"总共获取到 {self.dois_fetched} 个DOI")
        
        return all_dois
    
    def _extract_dois_from_xml(self, xml_content: str) -> List[str]:
        """从PubMed XML中提取DOI，只保留当前杂志编号+年份的DOI（如 s41551-023）"""
        dois = []
        journal_id = getattr(self, '_journal_id', '41551')
        year = getattr(self, '_year', 2023)
        # DOI 中年份为 3 位，如 2023 → 023
        year_in_doi = (year % 100)
        doi_prefix = f's{journal_id}-{year_in_doi:03d}'
        
        try:
            # 解析XML
            root = ET.fromstring(xml_content)
            
            # 查找所有ArticleId元素，类型为"doi"
            for article_id in root.findall('.//ArticleId'):
                if article_id.get('IdType', '').lower() == 'doi':
                    doi = article_id.text
                    if doi:
                        # 清理DOI格式
                        doi = doi.strip()
                        
                        # 移除可能的URL前缀
                        if doi.startswith('http'):
                            # 从URL中提取DOI部分
                            match = re.search(r'10\.\d+/[^\s]+', doi)
                            if match:
                                doi = match.group(0)
                        
                        # 只提取包含当前杂志编号+年份的DOI（如 s41551-023）
                        if doi_prefix in doi:
                            # 进一步清理，移除可能的后缀
                            doi = re.sub(r'[.,;:]$', '', doi)  # 移除末尾的标点
                            dois.append(doi)
                
            # 同时查找ELocationID元素
            for elocation in root.findall('.//ELocationID'):
                if elocation.get('EIdType', '').lower() == 'doi':
                    doi = elocation.text
                    if doi and doi_prefix in doi:
                        doi = re.sub(r'[.,;:]$', '', doi.strip())
                        if doi not in dois:  # 避免重复
                            dois.append(doi)
            
            # 记录找到的DOI数量
            if dois:
                logger.info(f"从XML中提取到 {len(dois)} 个包含'{doi_prefix}'的DOI")
                # 输出前几个DOI作为示例
                for i, doi in enumerate(dois[:5]):
                    logger.debug(f"DOI示例 {i+1}: {doi}")
                if len(dois) > 5:
                    logger.debug(f"... 还有 {len(dois)-5} 个DOI")
            
        except ET.ParseError as e:
            logger.error(f"XML解析失败: {e}")
            logger.error(f"XML内容前500字符: {xml_content[:500]}")
        except Exception as e:
            logger.error(f"提取DOI失败: {e}")
        
        return dois
    
    def doi_to_nature_pdf_url(self, doi: str) -> Tuple[Optional[str], Optional[str]]:
        """
        将DOI转换为Nature PDF下载链接和同行评审文件链接
        
        Args:
            doi: DOI号
            
        Returns:
            Tuple[文章PDF链接, 同行评审PDF链接] 或 (None, None)
        """
        if not doi:
            return None, None
        
        try:
            # 清理DOI
            doi = doi.strip()
            
            # 移除可能的URL前缀
            if doi.startswith('http'):
                # 提取DOI部分
                match = re.search(r'10\.\d+/[^\s/]+', doi)
                if match:
                    doi = match.group(0)
                else:
                    return None, None
            
            # Nature 系列 DOI 格式: 10.1038/s{杂志编号}-{年份3位如023}-xxxxx-x
            journal_id = getattr(self, '_journal_id', '41551')
            year = getattr(self, '_year', 2023)
            year_in_doi = year % 100  # 2023→23，格式化为 023
            pattern = rf'^s{re.escape(journal_id)}-{year_in_doi:03d}-\d{{5}}-[a-zA-Z0-9]{{1,2}}$'
            if '10.1038/' in doi:
                article_id = doi.replace('10.1038/', '')
                
                # 确保 article_id 符合当前杂志编号+年份的格式
                if re.match(pattern, article_id):
                    # 文章PDF链接
                    pdf_url = f"{self.NATURE_PDF_BASE}/{article_id}.pdf"
                    
                    # 获取同行评审文件链接
                    review_url = self._get_peer_review_url(article_id)
                    
                    return pdf_url, review_url
                else:
                    logger.debug(f"DOI格式不符合预期 (杂志{journal_id} {year}年): {doi}")
                    return None, None
            else:
                logger.debug(f"非 Nature 10.1038 DOI: {doi}")
                return None, None
            
        except Exception as e:
            logger.error(f"转换DOI失败: {doi} - {e}")
            return None, None

    def get_peer_review_url(self, article_id: str) -> Optional[str]:
        """
        获取Nature文章的同行评审文件链接
        
        Args:
            article_id: 文章ID (如 s41551-023-65288-9)
            
        Returns:
            同行评审PDF链接或None
        """
        try:
            # 构建文章页面URL
            article_url = f"{self.NATURE_PDF_BASE}/{article_id}"
            
            logger.debug(f"正在获取文章页面: {article_url}")
            
            # 获取文章页面
            time.sleep(self.delay)  # 添加延迟避免请求过快
            response = self.session.get(article_url, timeout=self.timeout)
            response.raise_for_status()
            
            html_content = response.text
            
            # 使用正则表达式查找同行评审文件链接
            # 查找包含 "transparent peer review file" 的链接
            # 模式匹配: <a ... href="...">Transparent Peer Review file</a>
            peer_review_patterns = [
                # 精确匹配用户提供的模式
                r'<a[^>]*class="print-link"[^>]*data-track-label="transparent peer review file"[^>]*href="([^"]+)"[^>]*>Transparent Peer Review file</a>',
                # 更通用的匹配模式
                r'<a[^>]*href="([^"]+)"[^>]*>[^<]*[Tt]ransparent [Pp]eer [Rr]eview [Ff]ile[^<]*</a>',
                # 匹配包含 "peer review" 的链接
                r'<a[^>]*href="([^"]+\.pdf)"[^>]*>[^<]*[Pp]eer [Rr]eview[^<]*</a>',
                # 匹配包含 "review" 的PDF链接
                r'<a[^>]*href="([^"]+)"[^>]*>[^<]*[Rr]eview[^<]*\.pdf[^<]*</a>',
            ]
            
            for pattern in peer_review_patterns:
                match = re.search(pattern, html_content, re.IGNORECASE)
                if match:
                    review_url = match.group(1)
                    
                    # 确保URL是完整的（如果是相对路径，则转换为绝对路径）
                    if review_url.startswith('/'):
                        review_url = f"https://www.nature.com{review_url}"
                    elif not review_url.startswith('http'):
                        # 如果是相对路径但没有前导斜杠
                        review_url = f"https://www.nature.com/{review_url}"
                    
                    logger.debug(f"找到同行评审文件链接: {review_url}")
                    return review_url
            
            # 如果没有找到，尝试查找包含 "peer review file" 的文本，然后查找附近的链接
            peer_review_text_pattern = r'[Pp]eer [Rr]eview [Ff]ile'
            if re.search(peer_review_text_pattern, html_content):
                # 查找附近的PDF链接
                pdf_links = re.findall(r'href="([^"]+\.pdf)"', html_content)
                for pdf_link in pdf_links:
                    if 'peer' in pdf_link.lower() or 'review' in pdf_link.lower():
                        # 确保URL是完整的
                        if pdf_link.startswith('/'):
                            pdf_link = f"https://www.nature.com{pdf_link}"
                        elif not pdf_link.startswith('http'):
                            pdf_link = f"https://www.nature.com/{pdf_link}"
                        
                        logger.debug(f"通过文本匹配找到可能的同行评审文件: {pdf_link}")
                        return pdf_link
            
            logger.debug(f"未找到文章 {article_id} 的同行评审文件链接")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"获取文章页面失败: {article_url} - {e}")
            return None
        except Exception as e:
            logger.error(f"解析同行评审链接失败: {e}")
            return None
    
    def _get_peer_review_url(self, article_id: str) -> Optional[str]:
        """
        获取Nature文章的同行评审文件链接（增强版）
        
        Args:
            article_id: 文章ID (如 s41551-023-65288-9)
            
        Returns:
            同行评审PDF链接或None
        """
        try:
            # 构建文章页面URL
            article_url = f"{self.NATURE_PDF_BASE}/{article_id}"
            
            logger.debug(f"正在获取文章页面: {article_url}")
            
            # 增强的请求头设置，模拟真实浏览器
            enhanced_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0',
                'DNT': '1',  # Do Not Track
            }
            
            # 更新session的headers
            original_headers = self.session.headers.copy()
            self.session.headers.update(enhanced_headers)
            
            try:
                # 增加延迟避免请求过快（Nature限制严格）
                # time.sleep(2.0)  # 增加到2秒
                
                # 获取文章页面
                response = self.session.get(article_url, timeout=1)
                
                # 检查响应状态
                if response.status_code == 406:
                    logger.warning(f"文章 {article_id} 触发406限制，尝试备用方法...")
                    # 尝试使用更保守的请求头
                    return None
                
                response.raise_for_status()
                
                html_content = response.text
                
                # 使用正则表达式查找同行评审文件链接
                # 查找包含 "transparent peer review file" 的链接
                peer_review_patterns = [
                    # 精确匹配用户提供的模式
                    r'<a[^>]*class="print-link"[^>]*data-track-label="transparent peer review file"[^>]*href="([^"]+)"[^>]*>Transparent Peer Review file</a>',
                    # 更通用的匹配模式
                    r'<a[^>]*href="([^"]+)"[^>]*>[^<]*[Tt]ransparent [Pp]eer [Rr]eview [Ff]ile[^<]*</a>',
                    # 匹配包含 "peer review" 的链接
                    r'<a[^>]*href="([^"]+\.pdf)"[^>]*>[^<]*[Pp]eer [Rr]eview[^<]*</a>',
                    # 匹配包含 "review" 的PDF链接
                    r'<a[^>]*href="([^"]+)"[^>]*>[^<]*[Rr]eview[^<]*\.pdf[^<]*</a>',
                ]
                
                for pattern in peer_review_patterns:
                    match = re.search(pattern, html_content, re.IGNORECASE)
                    if match:
                        review_url = match.group(1)
                        
                        # 确保URL是完整的（如果是相对路径，则转换为绝对路径）
                        if review_url.startswith('/'):
                            review_url = f"https://www.nature.com{review_url}"
                        elif not review_url.startswith('http'):
                            # 如果是相对路径但没有前导斜杠
                            review_url = f"https://www.nature.com/{review_url}"
                        
                        logger.debug(f"找到同行评审文件链接: {review_url}")
                        return review_url
                
                # 如果没有找到，尝试查找包含 "peer review file" 的文本，然后查找附近的链接
                # peer_review_text_pattern = r'[Pp]eer [Rr]eview [Ff]ile'
                # if re.search(peer_review_text_pattern, html_content):
                #     # 查找附近的PDF链接
                #     pdf_links = re.findall(r'href="([^"]+\.pdf)"', html_content)
                #     for pdf_link in pdf_links:
                #         if 'peer' in pdf_link.lower() or 'review' in pdf_link.lower():
                #             # 确保URL是完整的
                #             if pdf_link.startswith('/'):
                #                 pdf_link = f"https://www.nature.com{pdf_link}"
                #             elif not pdf_link.startswith('http'):
                #                 pdf_link = f"https://www.nature.com/{pdf_link}"
                            
                #             logger.debug(f"通过文本匹配找到可能的同行评审文件: {pdf_link}")
                #             return pdf_link
                
                logger.debug(f"未找到文章 {article_id} 的同行评审文件链接")
                return None
                
            finally:
                # 恢复原始headers
                self.session.headers.clear()
                self.session.headers.update(original_headers)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"获取文章页面失败: {article_url} - {e}")
            return None
        except Exception as e:
            logger.error(f"解析同行评审链接失败: {e}")
            return None

    def download_pdf(self, pdf_url: str, output_dir: str = "nature_pdfs_2023") -> bool:
        """
        下载PDF文件
        
        Args:
            pdf_url: PDF链接
            output_dir: 输出目录
            
        Returns:
            是否下载成功
        """
        if not pdf_url:
            return False
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 从URL提取文件名
        filename = pdf_url.split('/')[-1]
        if not filename.endswith('.pdf'):
            filename = f"{filename}.pdf"
        
        filepath = os.path.join(output_dir, filename)
        
        # 检查文件是否已存在
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            if file_size > 1023:  # 大于1KB认为有效
                logger.debug(f"文件已存在: {filename}")
                return True
        
        for attempt in range(self.max_retries):
            try:
                # 下载前延迟
                time.sleep(self.delay)
                
                # 下载文件
                response = self.session.get(pdf_url, stream=True, timeout=self.timeout)
                
                if response.status_code != 200:
                    if response.status_code == 404:
                        logger.debug(f"PDF不存在 (404): {pdf_url}")
                    elif response.status_code == 403:
                        logger.warning(f"访问被拒绝 (403): {pdf_url}")
                    else:
                        logger.debug(f"HTTP错误 {response.status_code}: {pdf_url}")
                    return False
                
                # 检查文件大小
                file_size = int(response.headers.get('content-length', 0))
                if file_size < 1023:  # 小于1KB，可能是错误页面
                    logger.warning(f"文件大小异常 ({file_size} 字节): {pdf_url}")
                    return False
                
                # 检查Content-Type
                content_type = response.headers.get('content-type', '').lower()
                if not ('pdf' in content_type or 'application/pdf' in content_type):
                    logger.warning(f"内容类型不是PDF: {content_type}")
                    return False
                
                # 保存文件
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 验证文件
                actual_size = os.path.getsize(filepath)
                if actual_size > 1023:
                    logger.info(f"下载成功: {filename} ({actual_size/1023/1023:.2f} MB)")
                    self.pdfs_downloaded += 1
                    return True
                else:
                    logger.warning(f"文件大小为0: {filename}")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return False
                    
            except requests.exceptions.Timeout:
                logger.warning(f"下载超时 ({attempt+1}/{self.max_retries}): {pdf_url}")
                if attempt < self.max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                continue
                
            except Exception as e:
                logger.error(f"下载失败: {pdf_url} - {e}")
                return False
        
        return False
    
    def save_results(self, dois: List[str], pdf_urls: List[str], 
                    output_file: str = None):
        """
        保存结果到文件
        
        Args:
            dois: DOI列表
            pdf_urls: PDF链接列表
            output_file: 输出文件名，默认根据杂志编号和年份生成
        """
        if output_file is None:
            journal_id = getattr(self, '_journal_id', '41551')
            year = getattr(self, '_year', 2023)
            output_file = f"nature_articles_{journal_id}_{year}.txt"
        journal_name = getattr(self, '_journal_name', 'Nature Biomedical Engineering')
        year = getattr(self, '_year', 2023)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {journal_name} {year} 年 Articles\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total Found: {self.total_found}\n")
            f.write(f"# DOIs Fetched: {self.dois_fetched}\n")
            f.write(f"# PDFs Downloaded: {self.pdfs_downloaded}\n")
            f.write(f"{'='*80}\n\n")
            
            for i, (doi, pdf_url) in enumerate(zip(dois, pdf_urls), 1):
                f.write(f"[{i}] DOI: {doi}\n")
                f.write(f"    PDF: {pdf_url}\n")
                f.write(f"{'-'*80}\n")
        
        logger.info(f"结果已保存到: {output_file}")
    
    def run(self, year: int = 2023, journal_name: str = "Nature Biomedical Engineering",
            journal_id: str = "41551", download_pdfs: bool = True, 
            test_mode: bool = False, output_dir: str = None):
        """
        主运行函数
        
        Args:
            year: 下载年份
            journal_name: 杂志名称（PubMed 期刊名）
            journal_id: 杂志编号（如 41551，对应 DOI 中的 s41551）
            download_pdfs: 是否下载PDF
            test_mode: 测试模式（只处理前10篇）
            output_dir: 输出目录，默认根据杂志编号和年份生成
        """
        self._journal_name = journal_name
        self._journal_id = str(journal_id)
        self._year = year
        if output_dir is None:
            output_dir_path = OUTPUTS_DIR / "downloads" / f"nature_pdfs_{self._journal_id}_{year}"
        else:
            output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_dir_path)
        
        logger.info("=" * 60)
        logger.info(f"{journal_name} ({journal_id}) {year} 年 文章下载器")
        logger.info("=" * 60)
        
        # 1. 搜索文章
        pmid_list = self.search_articles(year=year)
        if not pmid_list:
            logger.error("未找到文章，程序退出")
            return
        
        # 测试模式：只处理少量文章
        if test_mode:
            logger.info("⚠️  测试模式：只处理前10篇文章")
            pmid_list = pmid_list[:10]
        
        # 2. 获取DOI
        dois = self.fetch_dois_from_pmids(pmid_list)
        
        if not dois:
            logger.error("未获取到DOI，程序退出")
            return
        
        # 3. 转换为PDF链接
        logger.info("\n正在生成PDF下载链接...")
        pdf_urls = []
        review_urls = []
        valid_dois = []
        
        for doi in tqdm(dois, desc="生成PDF链接"):
            pdf_url, review_url = self.doi_to_nature_pdf_url(doi)
            if pdf_url and review_url:
                pdf_urls.append(pdf_url)
                review_urls.append(review_url)
                valid_dois.append(doi)
                logger.debug(f"DOI -> PDF: {doi} -> {os.path.basename(pdf_url)}")
        
        logger.info(f"成功生成 {len(pdf_urls)} 个PDF链接")
        
        # 4. 下载PDF（如果需要）
        if download_pdfs and pdf_urls:
            logger.info("\n开始下载PDF文件...")
            
            success_count = 0
            for pdf_url in tqdm(pdf_urls, desc="下载PDF"):
                if self.download_pdf(pdf_url, output_dir):
                    success_count += 1
                logger.info(f"PDF下载完成: {success_count}/{len(pdf_urls)} 成功")

            success_count = 0
            for review_url in tqdm(review_urls, desc="下载同行评审文件"):
                if self.download_pdf(review_url, output_dir):
                    success_count += 1
                logger.info(f"同行评审文件下载完成: {success_count}/{len(review_urls)} 成功")


        
        # 5. 保存结果
        output_file = output_dir_path / f"nature_articles_{self._journal_id}_{year}.txt"
        self.save_results(valid_dois, pdf_urls, output_file=str(output_file))
        
        # 6. 输出统计
        logger.info("=" * 60)
        logger.info("任务完成！")
        logger.info(f"总文章数: {self.total_found}")
        logger.info(f"获取DOI数: {self.dois_fetched}")
        logger.info(f"有效PDF链接: {len(pdf_urls)}")
        logger.info(f"下载PDF数: {self.pdfs_downloaded}")
        logger.info(f"文件保存至: {output_dir}/")
        logger.info("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='下载 Nature 系列杂志文章（可指定杂志名称、年份、杂志编号）')
    parser.add_argument('--journal-name', type=str, default='Nature Biomedical Engineering',
                        help='杂志名称（PubMed 期刊名，如 "Nature Biomedical Engineering" 或 "Nature communications"）')
    parser.add_argument('--year', type=int, default=2023, help='下载年份')
    parser.add_argument('--journal-id', type=str, default='41551',
                        help='杂志编号（如 41551 对应 Nature Biomedical Engineering，41467 对应 Nature Communications）')
    parser.add_argument('--email', type=str, default=getenv("NCBI_EMAIL"), help='您的邮箱（NCBI要求，也可设置 NCBI_EMAIL）')
    parser.add_argument('--api-key', type=str, default=getenv("NCBI_API_KEY"), help='NCBI API密钥（可选，也可设置 NCBI_API_KEY）')
    parser.add_argument('--no-download', action='store_true', help='只生成链接，不下载PDF')
    parser.add_argument('--test', action='store_true', default=False, help='测试模式（只处理前10篇）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='输出目录（默认: nature_pdfs_{杂志编号}_{年份}）')
    
    args = parser.parse_args()
    
    print("Nature 杂志文章下载器")
    print("=" * 50)
    print(f"杂志名称: {args.journal_name}")
    print(f"杂志编号: {args.journal_id}")
    print(f"年份: {args.year}")
    print(f"邮箱: {args.email}")
    print(f"API密钥: {'已提供' if args.api_key else '未提供'}")
    print(f"下载PDF: {'否' if args.no_download else '是'}")
    print(f"测试模式: {'是' if args.test else '否'}")
    default_output_dir = OUTPUTS_DIR / "downloads" / f"nature_pdfs_{args.journal_id}_{args.year}"
    print(f"输出目录: {args.output_dir or default_output_dir}")
    print()
    
    # 重要提示
    print("重要提示:")
    print("1. 请确保您的邮箱正确，NCBI可能会通过此邮箱联系")
    print("2. 如果没有API密钥，请求速率限制为每秒3次")
    print("3. 获取API密钥: https://www.ncbi.nlm.nih.gov/account/")
    print("4. 遵守NCBI使用条款: https://www.ncbi.nlm.nih.gov/home/about/policies/")
    print()
    
    # 创建下载器
    downloader = PubMedNatureDownloader(
        email=args.email,
        api_key=args.api_key
    )
    
    try:
        # 运行下载器
        downloader.run(
            year=args.year,
            journal_name=args.journal_name,
            journal_id=args.journal_id,
            download_pdfs=not args.no_download,
            test_mode=args.test,
            output_dir=args.output_dir
        )
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")
        logger.error(f"主程序出错: {e}", exc_info=True)


if __name__ == "__main__":
    # 安装依赖: pip install requests tqdm
    main()
