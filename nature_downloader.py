#!/usr/bin/env python3
"""
Nature Communications PDF爆破下载器 - 合并扫描和下载版本
直接尝试下载PDF，通过响应验证有效性
"""

import os
import re
import time
import random
import logging
import concurrent.futures
from datetime import datetime
from typing import List, Tuple, Optional
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'nature_direct_download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NatureDirectDownloader:
    """Nature Communications PDF直接下载器"""
    
    def __init__(self, base_url: str = "https://www.nature.com/articles"):
        """
        初始化下载器
        
        Args:
            base_url: Nature文章基础URL
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # 设置请求头，模拟浏览器
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-User': '?1',
            'Sec-Fetch-Dest': 'document',
        })
        
        # 请求参数
        self.min_delay = 0.3  # 最小延迟（秒）
        self.max_delay = 1.5  # 最大延迟（秒）
        self.timeout = 30     # 请求超时时间
        self.max_retries = 2  # 最大重试次数
        
        # 统计信息
        self.total_generated = 0
        self.total_tested = 0
        self.total_valid = 0
        self.total_downloaded = 0
        self.total_failed = 0
        
    def get_random_delay(self) -> float:
        """获取随机延迟时间"""
        return random.uniform(self.min_delay, self.max_delay)
    
    def generate_links(self, 
                      year_codes: List[str] = None,
                      number_range: Tuple[int, int] = (50000, 80000),
                      suffixes: List[str] = None) -> List[str]:
        """
        生成所有可能的PDF链接
        
        Args:
            year_codes: 年份代码列表，如 ['024', '025']
            number_range: 数字范围，如 (50000, 80000)
            suffixes: 后缀列表，如 ['0', '1', ..., '9', 'a', ..., 'z']
            
        Returns:
            生成的链接列表
        """
        if year_codes is None:
            year_codes = ['024', '025']  # 2024, 2025
        
        if suffixes is None:
            # 生成0-9和a-z
            suffixes = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('z')+1)]
        
        start_num, end_num = number_range
        links = []
        
        logger.info(f"开始生成链接...")
        logger.info(f"年份代码: {year_codes}")
        logger.info(f"数字范围: {start_num} - {end_num}")
        logger.info(f"后缀数量: {len(suffixes)}")
        
        # 计算总链接数
        total_links = len(year_codes) * (end_num - start_num + 1) * len(suffixes)
        logger.info(f"预计生成链接数: {total_links:,}")
        
        # 生成链接
        for year in year_codes:
            for num in range(start_num, end_num + 1):
                for suffix in suffixes:
                    # 格式: s41467-{year}-{num}-{suffix}.pdf
                    pdf_url = f"{self.base_url}/s41467-{year}-{num}-{suffix}.pdf"
                    links.append(pdf_url)
        
        self.total_generated = len(links)
        logger.info(f"实际生成链接数: {self.total_generated:,}")
        
        return links
    
    def try_download_pdf(self, pdf_url: str, output_dir: str = "nature_pdfs_direct") -> Tuple[bool, Optional[str], Optional[int]]:
        """
        尝试下载PDF文件，如果有效则保存
        
        Args:
            pdf_url: PDF链接
            output_dir: 输出目录
            
        Returns:
            (是否成功, 保存的文件路径, 文件大小)
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 从URL提取文件名
        filename = os.path.basename(urlparse(pdf_url).path)
        if not filename:
            filename = pdf_url.split('/')[-1]
        
        # 确保是.pdf扩展名
        if not filename.lower().endswith('.pdf'):
            filename = f"{filename}.pdf"
        
        filepath = os.path.join(output_dir, filename)
        
        # 检查文件是否已存在且有效
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            if file_size > 1024:  # 大于1KB认为有效
                logger.debug(f"文件已存在且有效: {filename} ({file_size/1024/1024:.2f} MB)")
                return True, filepath, file_size
        
        for attempt in range(self.max_retries):
            try:
                # 添加随机延迟
                time.sleep(self.get_random_delay())
                
                # 发送GET请求，设置stream=True以便分块下载
                response = self.session.get(
                    pdf_url, 
                    stream=True, 
                    timeout=self.timeout,
                    allow_redirects=True
                )
                
                # 检查HTTP状态码
                if response.status_code != 200:
                    if response.status_code == 404:
                        logger.debug(f"文件不存在 (404): {pdf_url}")
                    elif response.status_code == 403:
                        logger.warning(f"访问被拒绝 (403): {pdf_url}")
                    elif response.status_code == 429:
                        logger.warning(f"请求过多 (429)，等待后重试: {pdf_url}")
                        time.sleep(5 * (attempt + 1))
                        continue
                    else:
                        logger.debug(f"HTTP错误 {response.status_code}: {pdf_url}")
                    return False, None, None
                
                # 检查文件大小
                file_size = int(response.headers.get('content-length', 0))
                if file_size < 1024:  # 小于1KB，可能是错误页面
                    logger.warning(f"文件大小异常 ({file_size} 字节)，可能不是有效的PDF: {pdf_url}")
                    return False, None, None
                
                # 检查Content-Type
                content_type = response.headers.get('content-type', '').lower()
                if not ('pdf' in content_type or 'application/pdf' in content_type):
                    logger.warning(f"内容类型不是PDF: {content_type} - {pdf_url}")
                    return False, None, None
                
                # 保存文件
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # 验证保存的文件
                actual_size = os.path.getsize(filepath)
                if actual_size > 1024:
                    logger.info(f"下载成功: {filename} ({actual_size/1024/1024:.2f} MB)")
                    return True, filepath, actual_size
                else:
                    logger.warning(f"文件保存后大小为0: {filename}")
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return False, None, None
                    
            except requests.exceptions.Timeout:
                logger.warning(f"请求超时 ({attempt+1}/{self.max_retries}): {pdf_url}")
                if attempt < self.max_retries - 1:
                    wait_time = 3 * (attempt + 1)
                    time.sleep(wait_time)
                continue
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求异常 ({attempt+1}/{self.max_retries}): {pdf_url} - {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 * (attempt + 1)
                    time.sleep(wait_time)
                continue
                
            except Exception as e:
                logger.error(f"下载异常: {pdf_url} - {e}")
                return False, None, None
        
        return False, None, None
    
    def download_with_progress(self, 
                              links: List[str], 
                              output_dir: str = "nature_pdfs_direct",
                              max_workers: int = 5,
                              batch_size: int = 1000) -> dict:
        """
        下载所有链接，带进度显示
        
        Args:
            links: 链接列表
            output_dir: 输出目录
            max_workers: 最大并发数
            batch_size: 每批处理的链接数
            
        Returns:
            统计信息字典
        """
        logger.info(f"开始处理 {len(links):,} 个链接...")
        logger.info(f"输出目录: {output_dir}")
        logger.info(f"并发数: {max_workers}")
        logger.info(f"批次大小: {batch_size}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载已存在的文件记录
        existing_files = self._get_existing_files(output_dir)
        logger.info(f"找到 {len(existing_files)} 个已存在的PDF文件")
        
        # 统计数据
        stats = {
            'total': len(links),
            'skipped': 0,
            'success': 0,
            'failed': 0,
            'total_size_mb': 0
        }
        
        # 分批处理，避免内存占用过大
        total_batches = (len(links) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(links))
            batch_links = links[start_idx:end_idx]
            
            logger.info(f"\n处理批次 {batch_idx+1}/{total_batches}: 链接 {start_idx+1}-{end_idx}")
            
            # 使用进度条
            with tqdm(total=len(batch_links), desc=f"批次 {batch_idx+1}", unit="link") as pbar:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # 提交任务
                    future_to_url = {}
                    for url in batch_links:
                        # 检查是否已存在
                        filename = os.path.basename(urlparse(url).path)
                        if not filename.lower().endswith('.pdf'):
                            filename = f"{filename}.pdf"
                        filepath = os.path.join(output_dir, filename)
                        
                        if filepath in existing_files:
                            stats['skipped'] += 1
                            pbar.update(1)
                            continue
                        
                        future = executor.submit(self.try_download_pdf, url, output_dir)
                        future_to_url[future] = url
                    
                    # 处理结果
                    for future in concurrent.futures.as_completed(future_to_url):
                        url = future_to_url[future]
                        self.total_tested += 1
                        
                        try:
                            success, filepath, file_size = future.result()
                            
                            if success:
                                self.total_valid += 1
                                self.total_downloaded += 1
                                stats['success'] += 1
                                stats['total_size_mb'] += file_size / (1024 * 1024)
                                
                                # 添加到已存在文件集合
                                if filepath:
                                    existing_files.add(filepath)
                            else:
                                self.total_failed += 1
                                stats['failed'] += 1
                            
                        except Exception as e:
                            logger.error(f"处理链接时出错: {url} - {e}")
                            self.total_failed += 1
                            stats['failed'] += 1
                        
                        # 更新进度条
                        pbar.update(1)
                        pbar.set_postfix({
                            '成功': stats['success'],
                            '失败': stats['failed'],
                            '跳过': stats['skipped'],
                            '成功率': f"{stats['success']/max(self.total_tested, 1)*100:.2f}%"
                        })
            
            # 批次之间休息一下
            if batch_idx < total_batches - 1:
                delay = 10
                logger.info(f"批次完成，等待 {delay} 秒后继续下一批次...")
                time.sleep(delay)
        
        return stats
    
    def _get_existing_files(self, output_dir: str) -> set:
        """获取已存在的PDF文件集合"""
        existing_files = set()
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if filename.lower().endswith('.pdf'):
                    filepath = os.path.join(output_dir, filename)
                    file_size = os.path.getsize(filepath)
                    if file_size > 1024:  # 大于1KB认为有效
                        existing_files.add(filepath)
        return existing_files
    
    def run_bruteforce(self,
                      year_codes: List[str] = None,
                      number_range: Tuple[int, int] = (50000, 80000),
                      suffixes: List[str] = None,
                      test_mode: bool = True,
                      max_workers: int = 5,
                      batch_size: int = 1000,
                      output_dir: str = "nature_pdfs_direct") -> dict:
        """
        运行完整的爆破下载流程
        
        Args:
            year_codes: 年份代码列表
            number_range: 数字范围
            suffixes: 后缀列表
            test_mode: 测试模式（只测试少量链接）
            max_workers: 并发数
            batch_size: 每批处理的链接数
            output_dir: 输出目录
            
        Returns:
            统计信息字典
        """
        logger.info("=" * 60)
        logger.info("Nature Communications PDF直接下载器")
        logger.info("=" * 60)
        
        # 测试模式：只测试少量链接
        if test_mode:
            logger.info("⚠️  测试模式启用，只测试少量链接")
            if number_range[1] - number_range[0] > 100:
                number_range = (67970, 67980)  # 测试已知存在的范围
            if suffixes and len(suffixes) > 5:
                suffixes = ['0', '1', '2', '3', '4']  # 只测试5个后缀
        
        # 1. 生成链接
        all_links = self.generate_links(year_codes, number_range, suffixes)
        
        # 2. 直接下载（合并扫描和下载）
        stats = self.download_with_progress(
            all_links, 
            output_dir=output_dir,
            max_workers=max_workers,
            batch_size=batch_size
        )
        
        # 3. 汇总统计
        stats.update({
            'total_generated': self.total_generated,
            'total_tested': self.total_tested,
            'total_valid': self.total_valid,
            'total_downloaded': self.total_downloaded,
            'total_failed': self.total_failed,
            'success_rate': stats['success'] / max(self.total_tested, 1) * 100,
            'output_dir': output_dir
        })
        
        # 4. 输出报告
        logger.info("=" * 60)
        logger.info("下载完成！")
        logger.info(f"生成链接总数: {stats['total_generated']:,}")
        logger.info(f"测试链接数: {stats['total_tested']:,}")
        logger.info(f"跳过已存在: {stats['skipped']:,}")
        logger.info(f"成功下载: {stats['success']:,}")
        logger.info(f"下载失败: {stats['failed']:,}")
        logger.info(f"成功率: {stats['success_rate']:.2f}%")
        logger.info(f"总文件大小: {stats['total_size_mb']:.2f} MB")
        logger.info(f"平均文件大小: {stats['total_size_mb']/max(stats['success'], 1):.2f} MB")
        logger.info(f"PDF文件保存至: {stats['output_dir']}/")
        logger.info("=" * 60)
        
        return stats


def main():
    """主函数"""
    print("Nature Communications PDF直接下载器")
    print("=" * 50)
    print("基于链接规律: https://www.nature.com/articles/s41467-{年份}-{数字}-{后缀}.pdf")
    print("直接尝试下载，通过响应验证有效性")
    print()
    
    # 创建下载器
    downloader = NatureDirectDownloader()
    
    # 配置参数
    print("配置参数:")
    print("1. 年份代码 (如 024=2024, 025=2025)")
    print("2. 数字范围 (如 67970-67980)")
    print("3. 后缀 (0-9, a-z)")
    print()
    
    # 获取用户输入
    year_input = input("输入年份代码 (用逗号分隔，默认: 025): ").strip()
    if year_input:
        year_codes = [code.strip() for code in year_input.split(',')]
    else:
        year_codes = ['025']  # 默认只下载2025年
    
    range_input = input("输入数字范围 (格式: 开始-结束，默认: 67970-67980): ").strip()
    if range_input and '-' in range_input:
        start_str, end_str = range_input.split('-')
        number_range = (int(start_str.strip()), int(end_str.strip()))
    else:
        number_range = (67970, 67980)  # 测试范围
    
    suffix_input = input("输入后缀 (用逗号分隔，默认: 0): ").strip()
    if suffix_input:
        suffixes = []
        for item in suffix_input.split(','):
            item = item.strip()
            if '-' in item and len(item) == 3:  # 如 "0-9" 或 "a-z"
                start_char, end_char = item.split('-')
                if start_char.isdigit() and end_char.isdigit():
                    suffixes.extend([str(i) for i in range(int(start_char), int(end_char)+1)])
                elif start_char.isalpha() and end_char.isalpha():
                    suffixes.extend([chr(i) for i in range(ord(start_char), ord(end_char)+1)])
            else:
                suffixes.append(item)
    else:
        suffixes = ['0']  # 默认只测试后缀0
    
    # 并发设置
    workers_input = input("输入并发数 (默认: 3): ").strip()
    max_workers = int(workers_input) if workers_input.isdigit() else 3
    
    # 批次大小
    batch_input = input("输入批次大小 (默认: 500): ").strip()
    batch_size = int(batch_input) if batch_input.isdigit() else 500
    
    # 计算总链接数
    total_links = len(year_codes) * (number_range[1] - number_range[0] + 1) * len(suffixes)
    
    print(f"\n配置摘要:")
    print(f"  年份代码: {year_codes}")
    print(f"  数字范围: {number_range[0]} - {number_range[1]}")
    print(f"  后缀数量: {len(suffixes)}")
    print(f"  并发数: {max_workers}")
    print(f"  批次大小: {batch_size}")
    print(f"  预计链接数: {total_links:,}")
    
    if total_links > 10000:
        print(f"\n⚠️  警告: 链接数量较大 ({total_links:,})，可能需要较长时间！")
        print("建议先测试小范围验证规律。")
    
    # 运行模式选择
    print("\n运行模式:")
    print("1. 测试模式 (小范围验证)")
    print("2. 完整模式 (全范围下载)")
    
    mode = input("选择模式 (1/2，默认: 1): ").strip()
    
    if mode == '2':
        test_mode = False
        print("⚠️  完整模式将尝试所有链接，可能需要很长时间！")
    else:
        test_mode = True
        print("使用测试模式")
    
    # 确认
    print("\n" + "=" * 50)
    print("注意: 此方法会直接请求所有可能的PDF链接")
    print("可能会对服务器造成较大压力，请谨慎使用")
    confirm = input("是否开始? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("已取消")
        return
    
    try:
        # 运行下载器
        stats = downloader.run_bruteforce(
            year_codes=year_codes,
            number_range=number_range,
            suffixes=suffixes,
            test_mode=test_mode,
            max_workers=max_workers,
            batch_size=batch_size,
            output_dir="nature_pdfs_direct"
        )
        
        print(f"\n✅ 任务完成!")
        print(f"   总链接数: {stats['total_generated']:,}")
        print(f"   成功下载: {stats['success']:,}")
        print(f"   成功率: {stats['success_rate']:.2f}%")
        print(f"   文件保存至: {stats['output_dir']}/")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
    except Exception as e:
        print(f"\n❌ 出错: {e}")
        logger.error(f"主程序出错: {e}", exc_info=True)


if __name__ == "__main__":
    # 安装进度条库: pip install tqdm
    main()