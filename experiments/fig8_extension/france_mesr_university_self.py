"""Corroborate exact French university self names from ministry directories."""
from __future__ import annotations

from .uk_devolved_university_self import run

SOURCES = {
    'doctorate_directory': (
        'https://www.enseignementsup-recherche.gouv.fr/fr/etablissements-d-enseignement-superieur-accredites-delivrer-le-doctorat-46523',
        'Aix-Marseille Université|Avignon Université|CY Cergy Paris Université|'
        'Institut Polytechnique de Paris|La Rochelle Université|Le Mans Université|'
        'Sorbonne Université|Université Bordeaux Montaigne|Université Claude Bernard Lyon 1|'
        'Université Clermont Auvergne|Université Côte d\'Azur|Université d\'Angers|'
        'Université d\'Artois|Université d\'Evry Val d\'Essonne|Université d\'Orléans|'
        'Université de Bordeaux|Université de Bourgogne|Université de Bretagne Occidentale|'
        'Université de Bretagne Sud|Université de Caen Normandie|'
        'Université de Corse Pasquale Paoli|Université de Franche-Comté|'
        'Université de Haute-Alsace|Université de Lille|Université de Lorraine|'
        'Université de Montpellier|Université de Nantes|Université de Nîmes|'
        'Université de Pau et des Pays de l\'Adour|Université de Perpignan - Via Domitia|'
        'Université de Picardie Jules Verne|Université de Poitiers|'
        'Université de Reims Champagne-Ardenne|Université de Rennes 1|'
        'Université de Rouen Normandie|Université de Strasbourg|'
        'Université de technologie de Belfort Montbéliard|Université de technologie de Compiègne|'
        'Université de technologie de Troyes|Université de Toulon|Université de Tours|'
        'Université de Versailles-Saint-Quentin en Yvelines|Université du Littoral Côte d\'Opale|'
        'Université Grenoble Alpes|Université Gustave Eiffel|Université Jean Monnet Saint-Etienne|'
        'Université Jean Moulin Lyon 3|Université Le Havre Normandie|Université Lumière - Lyon 2|'
        'Université Panthéon-Assas|Université Paris 1 Panthéon Sorbonne|'
        'Université Paris 8 - Vincennes - Saint-Denis|Université Paris Nanterre|'
        'Université Paris Sciences et Lettres|Université Paris-Est Créteil Val-de-Marne|'
        'Université Paris-Saclay|Université Paul-Valéry - Montpellier 3|'
        'Université Polytechnique Hauts-de-France|Université Rennes 2|Université Savoie Mont Blanc|'
        'Université Sorbonne Nouvelle - Paris 3|Université Sorbonne Paris Nord|'
        'Université Toulouse-I Capitole|Université Toulouse-II Jean Jaurès|'
        'Université Toulouse-III Paul Sabatier'),
    'ile_de_france_2024': (
        'https://www.enseignementsup-recherche.gouv.fr/fr/les-etablissements-d-enseignement-superieur-et-les-organismes-de-recherche-en-ile-de-france-45988',
        'Université Paris Cité'),
}


if __name__ == '__main__':
    run(SOURCES, 'france_mesr_self',
        {'doctorate_directory': 'FR', 'ile_de_france_2024': 'FR'},
        'Exact signed entity self only. No university component, hospital, joint-center, '
        'federation-member or historical rename allocation; no overseas country assignment. '
        'The older doctorate directory corroborates institution identity, not accreditation '
        'throughout 2023–2025. Unmatched punctuation, accents and translated names are held.')
