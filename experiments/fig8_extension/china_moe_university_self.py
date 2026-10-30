"""Reuse MOE's named subordinate universities for exact primary signatures."""
from __future__ import annotations

from .uk_devolved_university_self import run

SOURCES = {
    'china': ('https://en.moe.gov.cn/subordinate_universities/',
        'Peking University|Tsinghua University|Renmin University of China|'
        'Beijing Normal University|China Agricultural University|'
        'Beijing Foreign Studies University|University of Science and Technology Beijing|'
        'Beijing Language and Culture University|Beijing University of Chemical Technology|'
        'Beijing Jiaotong University|Beijing University of Posts and Telecommunications|'
        'Beijing Forestry University|Communication University of China|'
        'Central University of Finance and Economics|Central Conservatory of Music|'
        'China University of Political Science and Law|The Central Academy of Drama|'
        'China Central Academy of Fine Arts|Beijing University of Chinese Medicine|'
        'University of International Business and Economics|Tianjin University|Nankai University|'
        'Dalian University of Technology|Jilin University|Northeast Normal University|'
        'Northeast Forestry University|Fudan University|Shanghai Jiao Tong University|'
        'Tongji University|East China University of Science and Technology|Donghua University|'
        'East China Normal University|Shanghai University of Finance and Economics|'
        'Shanghai International Studies University|Nanjing University|Southeast University|'
        'Hohai University|Jiangnan University|Nanjing Agricultural University|'
        'China Pharmaceutical University|Zhejiang University|Hefei University of Technology|'
        'Xiamen University|Shandong University|Ocean University of China|Wuhan University|'
        'Huazhong University of Science and Technology|Wuhan University of Technology|'
        'Central China Normal University|Huazhong Agricultural University|'
        'Central South University|Hunan University|Sun Yat-sen University|'
        'South China University of Technology|Chongqing University|Southwest University|'
        'Sichuan University|Southwestern University of Finance and Economics|'
        'Southwest Jiaotong University|University of Electronic Science and Technology of China|'
        'Northwest A&F University|Xidian University|Shaanxi Normal University|Lanzhou University'),
}


if __name__ == '__main__':
    run(SOURCES, 'china_moe_self', {'china': 'CN'},
        'MOE subordinate-university names read from indexed official page text; '
        'direct web open reports redirect loop. Exact self names only, corroborated '
        'by primary signatures dated 2023–2025 and unique existing education entity. '
        'Excludes ambiguous shared-name petroleum, geosciences and mining universities, '
        'North China Electric Power branch distinction, globally ambiguous Northeastern '
        'University, and name punctuation crosswalks. Hospitals, subsidiaries, '
        'departments, joint centers and branch campuses are not inferred. This is '
        'not the national complete university list; omitted schools remain in scope. '
        'Current listing does not establish historical subsidiary ownership. '
        'Already reviewed entities, signed countries and Nature records preserved.',
        evidence_access='Official page text returned by web search; direct web open failed with redirect loop')
