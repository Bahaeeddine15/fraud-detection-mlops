// Jenkinsfile — Pipeline CI/CD pour le projet de détection de fraude
//
// Étapes : préparer les données -> tester le code -> ré-entraîner le
// modèle -> construire l'image Docker.
// Chaque stage "agent { docker {...} }" lance un conteneur EPHÉMÈRE
// (créé pour la durée du stage, puis détruit) avec Python déjà installé,
// plutôt que d'installer Python directement dans Jenkins.
// "reuseNode: true" fait que ce conteneur éphémère partage le même
// dossier de travail que Jenkins, donc les fichiers créés à un stage
// (comme les .joblib) sont visibles au stage suivant.

pipeline {
    agent any

    stages {

        stage('Préparer les données') {
            // Ce stage tourne directement sur l'agent Jenkins (pas dans
            // un conteneur Python), car c'est lui qui a accès au volume
            // /data-source monté au lancement du conteneur Jenkins.
            steps {
                sh '''
                    mkdir -p data
                    cp /data-source/creditcard.csv data/creditcard.csv
                '''
            }
        }

        stage('Installer les dépendances et tester') {
            agent {
                docker {
                    image 'python:3.11-slim'
                    reuseNode true
                }
            }
            steps {
                // On crée un environnement virtuel DANS le workspace
                // (toujours accessible en écriture, contrairement au
                // dossier home de l'utilisateur du conteneur éphémère,
                // qui a causé l'erreur "Permission denied: '/.local'").
                sh '''
                    python -m venv .venv
                    . .venv/bin/activate
                    pip install --no-cache-dir -r requirements.txt
                    pytest tests/ -v
                '''
            }
        }

        stage('Ré-entraîner le modèle (candidat)') {
            agent {
                docker {
                    image 'python:3.11-slim'
                    reuseNode true
                }
            }
            steps {
                // Si train.py échoue (AUC-PR <= 0.80, garde-fou absolu),
                // ce stage échoue et le pipeline s'arrête ici.
                sh '''
                    . .venv/bin/activate
                    python src/train.py
                '''
            }
        }

        stage('Comparer et promouvoir (F9)') {
            agent {
                docker {
                    image 'python:3.11-slim'
                    reuseNode true
                }
            }
            steps {
                script {
                    // returnStatus: true capture le code de sortie SANS
                    // faire échouer le stage — on veut décider nous-mêmes
                    // quoi faire si le candidat est rejeté (pas planter
                    // le pipeline, juste sauter les stages suivants).
                    def exitCode = sh(
                        script: '''
                            . .venv/bin/activate
                            python src/compare_and_promote.py
                        ''',
                        returnStatus: true
                    )
                    // Variable d'environnement lue par les stages suivants
                    // via leur condition "when".
                    env.PROMOTED = (exitCode == 0) ? "true" : "false"
                }
            }
        }

        stage('Construire l\'image Docker') {
            // Ne s'exécute QUE si le candidat a été promu à l'étape
            // précédente. C'est le coeur de F9 : pas de nouveau build/
            // déploiement si le modèle n'est pas meilleur que la prod.
            when {
                environment name: 'PROMOTED', value: 'true'
            }
            steps {
                sh 'docker build -t fraude-api:${BUILD_NUMBER} .'
                sh 'docker tag fraude-api:${BUILD_NUMBER} fraude-api:latest'
            }
        }
    }

    post {
        success {
            echo 'Pipeline terminé avec succès : tests passés, modèle entraîné, image construite.'
        }
        failure {
            echo 'Pipeline en échec — vérifier les logs du stage concerné.'
        }
    }
}