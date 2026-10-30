"""Corroborate exact signed university names with the NTA central list."""
from __future__ import annotations

from .uk_devolved_university_self import run

SOURCES = {
    'india': ('https://cuet.nta.nic.in/universities/participating-central-universities/',
        'Aligarh Muslim University|Assam University|Babasaheb Bhimrao Ambedkar University|'
        'Banaras Hindu University|Central Sanskrit University|'
        'Central Tribal University of Andhra Pradesh|Central University of Andhra Pradesh|'
        'Central University of Gujarat|Central University of Haryana|'
        'Central University of Himachal Pradesh|Central University of Jammu|'
        'Central University of Jharkhand|Central University of Karnataka|'
        'Central University of Kashmir|Central University of Kerala|Central University of Odisha|'
        'Central University of Punjab|Central University of Rajasthan|'
        'Central University of South Bihar|Central University of Tamil Nadu|'
        'Dr. Harisingh Gour Vishwavidyalaya|Guru Ghasidas Vishwavidyalaya|'
        'Hemvati Nandan Bahuguna Garhwal University|Jamia Millia Islamia|'
        'Jawaharlal Nehru University|Mahatma Gandhi Antarrashtriya Hindi Vishwavidyalaya|'
        'Mahatma Gandhi Central University|Manipur University|Maulana Azad National Urdu University|'
        'Mizoram University|Nagaland University|North-Eastern Hill University|'
        'Pondicherry University|Rajiv Gandhi National Aviation University|'
        'Shri Lal Bahadur Shastri National Sanskrit University|Sikkim University|'
        'South Asian University|Tezpur University|The English and Foreign Languages University|'
        'Tripura University|University of Allahabad|University of Delhi|'
        'University of Hyderabad|Visva-Bharati University'),
}


if __name__ == '__main__':
    run(SOURCES, 'india_central_self', {'india': 'IN'},
        'NTA government CUET participating central-university list; exact names '
        'corroborate university-self identity only alongside the existing primary '
        '2023–2025 paper signature and unique education entity. Not a complete '
        'India university list or accreditation adjudication. Excludes IIM '
        'Sambalpur, newer Sammakka Sarakka institution, renamed National Sanskrit '
        'entry and entries with additional location qualifiers. No predecessor '
        'or subsidiary inference. Hospitals, departments, colleges, joint centers '
        'and branch campuses not allocated. Countries and existing Nature records '
        'preserved; all nonmatching names remain held.')
