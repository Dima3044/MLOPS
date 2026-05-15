pipeline {
    agent any
    environment {
        VENV = "${env.WORKSPACE}/ml_env"
        PYTHON = "${env.WORKSPACE}/ml_env/bin/python"
        PIP = "${env.WORKSPACE}/ml_env/bin/pip"
        MLFLOW_TRACKING_URI = "${env.WORKSPACE}/mlruns"
    }
    stages {
        stage('📦 Setup Environment') {
            steps {
                sh "python3 -m venv ${VENV}"
                sh "${PIP} install --upgrade pip setuptools wheel"
                sh "${PIP} install -r requirements.txt"
            }
        }
        stage('📥 Download & Preprocess') {
            steps {
                sh "${PYTHON} download.py"
            }
        }
        stage('🧠 Train & Log Model') {
            steps {
                sh "${PYTHON} train_model.py"
            }
        }
        stage('🚀 Deploy Service') {
            steps {
                sh '''
                    export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}
                    export BUILD_ID=dontKillMe
                    export JENKINS_NODE_COOKIE=dontKillMe
                    
                    PATH_MODEL=$(cat best_model.txt)
                    echo "🌐 Запуск MLflow serve на порту 5003..."
                    mlflow models serve -m "$PATH_MODEL" -p 5003 --no-conda &
                    sleep 8
                '''
            }
        }
        stage('🩺 Health Check') {
            steps {
                sh '''
                    echo "🔍 Проверка доступности сервиса..."
                    # 11 признаков: 9 числовых + 2 dummy (sex_F, sex_M)
                    curl -f -X POST http://127.0.0.1:5003/invocations \
                      -H "Content-Type: application/json" \
                      -d '{"inputs": [[0.45, 0.30, 0.09, 0.15, 0.10, 0.05, 0.05, 0.38, 0.90, 1.0, 0.0]]}' \
                      || (echo "❌ Health check failed!" && exit 1)
                    echo "✅ Сервис успешно отвечает!"
                '''
            }
        }
    }
    post {
        always {
            sh 'pkill -f "mlflow models serve" || true'
        }
    }
}