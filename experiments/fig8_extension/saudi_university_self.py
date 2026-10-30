"""Reuse Saudi education ministry's English university names for exact self signatures."""
from __future__ import annotations

from .uk_devolved_university_self import run

SOURCES = {
    'saudi_arabia': ('https://object.moe.gov.sa/studyinsaudi/Profile_en-US.pdf',
        'Shaqra University|King Faisal University|Jeddah University|Umm Al-Qura University|Prince Sattam Bin Abdulaziz University|Taif University|King Saud University|Islamic University of Madinah|Northern Border University|Najran University|Taibah University|Princess Nourah University|King Abdulaziz University|Jazan University|Tabuk University|Imam Abdulrahman Bin Faisal University|Qassim University|Hafr Al Batin University|Jouf University|Majmaah University|Bisha University|Al-Baha University|Saudi Electronic University|King Khalid University'),
}

if __name__ == '__main__':
    run(SOURCES, 'saudi_self', {'saudi_arabia': 'SA'},
        'Official ministry English brochure PDF page 8, Saudi Universities grid. Exact names only, corroborated by original 2023–2025 declared university signatures. Incomplete Imam Mohammad label and typographic Hail alias not expanded. Nonmatching registry names held. No inferred historical subsidiary ownership, hospitals, joint centers, branch campuses or countries; not a claim that all Saudi universities are listed.')
