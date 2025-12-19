import numpy as np
import re

def load_control_matrix_from_file(filename='reference.txt'):
    """
    Lit la matrice de contrôle située entre les balises #BEGIN et #END.
    Chaque retour à la ligne dans le fichier correspond à une ligne de la matrice.
    """
    try:
        with open(filename, 'r') as file:
            content = file.read()

        # Utilisation d'une expression régulière pour extraire le bloc entre les balises
        # re.DOTALL permet au point (.) de correspondre aussi aux retours à la ligne
        pattern = r"#BEGIN matrice de controle\s*(.*?)\s*#END matrice de controle"
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            print(f"⚠️ Aucune matrice de contrôle trouvée dans {filename}")
            return None

        # Récupération du texte brut de la matrice
        matrix_text = match.group(1).strip()
        
        control_matrix = []
        
        # On découpe par ligne
        lines = matrix_text.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                # On découpe chaque ligne par les virgules et on convertit en float
                row = [float(val) for val in line.split(',') if val.strip()]
                control_matrix.append(row)

        # Conversion en tableau NumPy pour faciliter les calculs d'OA
        return np.array(control_matrix)

    except Exception as e:
        print(f"❌ Erreur lors de la lecture de la matrice de contrôle : {e}")
        return None

def load_MI_from_file(filename='reference.txt'):
    """
    Lit la matrice de contrôle située entre les balises #BEGIN et #END.
    Chaque retour à la ligne dans le fichier correspond à une ligne de la matrice.
    """
    try:
        with open(filename, 'r') as file:
            content = file.read()

        # Utilisation d'une expression régulière pour extraire le bloc entre les balises
        # re.DOTALL permet au point (.) de correspondre aussi aux retours à la ligne
        pattern = r"#BEGIN MI\s*(.*?)\s*#END MI"
        match = re.search(pattern, content, re.DOTALL)

        if not match:
            print(f"⚠️ Aucune MI trouvée dans {filename}")
            return None

        # Récupération du texte brut de la matrice
        matrix_text = match.group(1).strip()
        
        MI = []
        
        # On découpe par ligne
        lines = matrix_text.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                # On découpe chaque ligne par les virgules et on convertit en float
                row = [float(val) for val in line.split(',') if val.strip()]
                MI.append(row)

        # Conversion en tableau NumPy pour faciliter les calculs d'OA
        return np.array(MI)

    except Exception as e:
        print(f"❌ Erreur lors de la lecture de MI : {e}")
        return None
    
MC = load_control_matrix_from_file()
MI = load_MI_from_file()
v=MI[0,:]

V=110*v @ MC
print(V)