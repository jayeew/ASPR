import os
import time
import re
import requests
from urllib.parse import urlparse, urljoin
from pathlib import Path
from typing import Optional, Dict, Any
import random

class ACLPDFDownloader:
    """
    专门针对ACL Anthology网站的PDF下载器
    采用先访问HTML页面再下载PDF的策略绕过反爬机制
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        """
        初始化下载器
        
        参数:
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self._setup_session()
    
    def _setup_session(self):
        """设置会话的默认请求头"""
        # 随机选择一个User-Agent
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
        ]
        
        self.session.headers.update({
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        })
    
    def _extract_pdf_url_from_html(self, html_url: str) -> Optional[str]:
        """
        从HTML页面提取PDF下载链接
        
        参数:
            html_url: HTML页面URL
            
        返回:
            PDF下载URL，如果提取失败则返回None
        """
        try:
            response = self.session.get(html_url, timeout=10)
            response.raise_for_status()
            
            # 在HTML中查找PDF链接
            html_content = response.text
            
            # 方法1: 查找a标签中的PDF链接
            pdf_patterns = [
                r'href="([^"]+\.pdf)"',  # href="xxx.pdf"
                r'href=[\'"]([^\'"]+\.pdf)[\'"]',  # href='xxx.pdf'
                r'https://[^"\']+\.pdf',  # 直接匹配PDF URL
            ]
            
            for pattern in pdf_patterns:
                matches = re.findall(pattern, html_content, re.IGNORECASE)
                for match in matches:
                    if 'aclweb.org' in match or 'aclanthology.org' in match:
                        if match.startswith('http'):
                            return match
                        else:
                            return urljoin(html_url, match)
            
            # 方法2: 查找meta标签中的PDF链接
            meta_pattern = r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']'
            meta_match = re.search(meta_pattern, html_content, re.IGNORECASE)
            if meta_match:
                return meta_match.group(1)
            
            return None
            
        except Exception as e:
            print(f"从HTML提取PDF链接失败: {e}")
            return None
    
    def download_acl_pdf(self, pdf_url: str, save_dir: str = ".", 
                        filename: Optional[str] = None) -> str:
        """
        下载ACL Anthology的PDF文件
        
        参数:
            pdf_url: PDF文件的URL地址
            save_dir: 保存目录，默认为当前目录
            filename: 保存的文件名，如果为None则从URL中提取
            
        返回:
            保存的文件路径
        """
        # 创建保存目录
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # 从URL中提取文件名
        if filename is None:
            parsed_url = urlparse(pdf_url)
            filename = os.path.basename(parsed_url.path)
            if not filename or not filename.lower().endswith('.pdf'):
                filename = "acl_paper.pdf"
        
        save_path = os.path.join(save_dir, filename)
        
        print(f"目标PDF URL: {pdf_url}")
        print(f"保存到: {save_path}")
        
        # 1. 构造HTML页面URL
        # ACL Anthology的URL格式: https://aclanthology.org/2020.acl-main.447.pdf
        # 对应的HTML页面: https://aclanthology.org/2020.acl-main.447/
        html_url = pdf_url.replace('.pdf', '/')
        if not html_url.endswith('/'):
            html_url += '/'
        
        print(f"1. 访问HTML页面: {html_url}")
        
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    print(f"第 {attempt} 次重试...")
                    time.sleep(self.retry_delay * attempt)
                
                # 2. 先访问HTML页面，建立会话
                html_headers = self.session.headers.copy()
                html_headers.update({
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Referer': 'https://aclanthology.org/',
                })
                
                html_response = self.session.get(
                    html_url,
                    headers=html_headers,
                    timeout=15,
                    allow_redirects=True
                )
                html_response.raise_for_status()
                
                print(f"HTML页面访问成功 (状态码: {html_response.status_code})")
                
                # 3. 从HTML页面提取PDF链接（如果需要）
                extracted_pdf_url = self._extract_pdf_url_from_html(html_url)
                if extracted_pdf_url and extracted_pdf_url != pdf_url:
                    print(f"从HTML页面提取到新的PDF URL: {extracted_pdf_url}")
                    pdf_url = extracted_pdf_url
                
                # 4. 等待随机时间，模拟用户行为
                wait_time = random.uniform(1.0, 3.0)
                print(f"等待 {wait_time:.1f} 秒后下载PDF...")
                time.sleep(wait_time)
                
                # 5. 下载PDF文件
                pdf_headers = self.session.headers.copy()
                pdf_headers.update({
                    'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Encoding': 'identity',  # 避免压缩，确保正确获取文件大小
                    'Referer': html_url,  # 重要：设置Referer为HTML页面
                    'Upgrade-Insecure-Requests': '0',
                })
                
                print(f"2. 下载PDF文件: {pdf_url}")
                response = self.session.get(
                    pdf_url,
                    headers=pdf_headers,
                    stream=True,
                    timeout=30
                )
                
                # 检查响应状态
                if response.status_code == 200:
                    print(f"PDF请求成功 (状态码: {response.status_code})")
                    
                    # 获取文件大小
                    total_size = int(response.headers.get('content-length', 0))
                    
                    # 检查Content-Type
                    content_type = response.headers.get('Content-Type', '').lower()
                    if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                        print(f"警告: Content-Type不是PDF: {content_type}")
                        # 但仍然继续下载，有些服务器可能返回错误的Content-Type
                    
                    # 下载文件
                    with open(save_path, 'wb') as file:
                        if total_size == 0:
                            file.write(response.content)
                            print("下载完成（无进度信息）")
                        else:
                            downloaded_size = 0
                            chunk_size = 8192
                            
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    file.write(chunk)
                                    downloaded_size += len(chunk)
                                    
                                    progress = (downloaded_size / total_size) * 100
                                    print(f"\r下载进度: {progress:.1f}% ({downloaded_size}/{total_size} bytes)", end="")
                            
                            print("\n下载完成")
                    
                    # 验证文件
                    if os.path.exists(save_path):
                        file_size = os.path.getsize(save_path)
                        if file_size > 0:
                            print(f"文件已成功保存，大小: {file_size:,} bytes")
                            return save_path
                        else:
                            raise Exception("下载的文件大小为0")
                    else:
                        raise Exception("文件保存失败")
                        
                elif response.status_code == 403:
                    print("403禁止访问，可能需要更新User-Agent或等待更长时间")
                    continue
                elif response.status_code == 404:
                    raise Exception("404文件未找到")
                elif response.status_code == 409:
                    print("409冲突错误，尝试不同的策略...")
                    # 尝试使用更长的延迟
                    time.sleep(self.retry_delay * 2)
                    continue
                else:
                    raise Exception(f"HTTP错误 {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"请求异常: {e}")
                if attempt < self.max_retries - 1:
                    continue
                else:
                    raise Exception(f"下载失败: {e}")
            except Exception as e:
                print(f"错误: {e}")
                if attempt < self.max_retries - 1:
                    continue
                else:
                    raise Exception(f"下载失败: {e}")
        
        raise Exception(f"下载失败，已达到最大重试次数 {self.max_retries}")
    
    def download_pdf_direct(self, pdf_url: str, save_path: str) -> bool:
        """
        直接下载PDF（备用方法）
        
        参数:
            pdf_url: PDF文件的URL
            save_path: 保存路径
            
        返回:
            成功返回True，失败返回False
        """
        try:
            # 使用简单的requests直接下载
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': '*/*',
                'Accept-Encoding': 'identity',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(pdf_url, headers=headers, timeout=30, stream=True)
            
            if response.status_code == 200:
                with open(save_path, 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                return True
            else:
                print(f"直接下载失败，状态码: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"直接下载异常: {e}")
            return False
    
    def close(self):
        """关闭会话"""
        self.session.close()

def test_acl_download():
    """测试ACL PDF下载"""
    print("=" * 60)
    print("测试ACL Anthology PDF下载")
    print("=" * 60)
    
    # 创建测试目录
    test_dir = "./downloads"
    Path(test_dir).mkdir(exist_ok=True)
    
    # 测试URLs
    test_urls = [
        "https://aclanthology.org/2020.acl-main.447.pdf",  # 用户提供的URL
        # "https://aclanthology.org/2021.acl-long.100.pdf",  # 另一个ACL论文
        # "https://aclanthology.org/2022.acl-long.1.pdf",    # 再一个ACL论文
    ]
    
    # 创建下载器
    downloader = ACLPDFDownloader(max_retries=2, retry_delay=3.0)
    
    try:
        for i, url in enumerate(test_urls):
            print(f"\n测试 {i+1}: {url}")
            print("-" * 40)
            
            try:
                # 使用方法1: 先访问HTML页面再下载
                filename = f"acl_paper_{i+1}.pdf"
                save_path = os.path.join(test_dir, filename)
                
                saved_file = downloader.download_acl_pdf(
                    url,
                    save_dir=test_dir,
                    filename=filename
                )
                
                if saved_file and os.path.exists(saved_file):
                    file_size = os.path.getsize(saved_file)
                    print(f"✓ 下载成功: {saved_file} ({file_size:,} bytes)")
                else:
                    print("✗ 下载失败")
                    
            except Exception as e:
                print(f"✗ 下载失败: {e}")
                
                # 尝试备用方法
                print("尝试备用方法...")
                if downloader.download_pdf_direct(url, save_path):
                    file_size = os.path.getsize(save_path)
                    print(f"✓ 备用方法下载成功: {save_path} ({file_size:,} bytes)")
                else:
                    print("✗ 备用方法也失败")
    
    finally:
        downloader.close()
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)

if __name__ == "__main__":
    # 运行测试
    test_acl_download()
