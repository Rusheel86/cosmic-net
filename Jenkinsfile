pipeline {
    agent any

    environment {
        ACR_LOGIN_SERVER = 'cosmicnetacr.azurecr.io'
        ACR_USERNAME     = 'cosmicnetacr'
    }

    stages {

        stage('Checkout') {
            steps {
                echo '=== Pulling latest code from GitHub ==='
                checkout scm
            }
        }

        stage('Login to ACR') {
            steps {
                echo '=== Logging into Azure Container Registry ==='
                withCredentials([string(credentialsId: 'ACR_PASSWORD', variable: 'ACR_PASS')]) {
                    sh '''
                        echo $ACR_PASS | docker login $ACR_LOGIN_SERVER \
                            --username $ACR_USERNAME \
                            --password-stdin
                        echo "ACR login successful"
                    '''
                }
            }
        }

        stage('Build Backend Image') {
            steps {
                echo '=== Copying model artifact ==='
                sh '''
                    cp /Users/nealsalian/Desktop/cosmic-net/backend/model_artifacts/best_model_augmented.pt \
                       ./backend/model_artifacts/best_model_augmented.pt
                '''
                echo '=== Building Backend Docker Image ==='
                sh '''
                    docker buildx build \
                        --platform linux/amd64 \
                        --tag $ACR_LOGIN_SERVER/cosmic-net-backend:latest \
                        --tag $ACR_LOGIN_SERVER/cosmic-net-backend:$BUILD_NUMBER \
                        --push \
                        ./backend
                    echo "Backend image pushed successfully"
                '''
            }
        }

        stage('Build Frontend Image') {
            steps {
                echo '=== Building Frontend Docker Image ==='
                sh '''
                    docker buildx build \
                        --platform linux/amd64 \
                        --tag $ACR_LOGIN_SERVER/cosmic-net-frontend:latest \
                        --tag $ACR_LOGIN_SERVER/cosmic-net-frontend:$BUILD_NUMBER \
                        --push \
                        ./frontend
                    echo "Frontend image pushed successfully"
                '''
            }
        }

        stage('Verify Images in ACR') {
            steps {
                echo '=== Verifying images pushed to Azure Container Registry ==='
                withCredentials([string(credentialsId: 'ACR_PASSWORD', variable: 'ACR_PASS')]) {
                    sh '''
                        echo "========================================"
                        echo "COSMIC-NET CI PIPELINE SUCCESSFUL"
                        echo "Build Number : #$BUILD_NUMBER"
                        echo "Backend  : $ACR_LOGIN_SERVER/cosmic-net-backend:$BUILD_NUMBER"
                        echo "Frontend : $ACR_LOGIN_SERVER/cosmic-net-frontend:$BUILD_NUMBER"
                        echo "Images stored in Azure Container Registry"
                        echo "========================================"
                    '''
                }
            }
        }
    }

    post {
        success {
            echo 'Pipeline completed — both images built and pushed to ACR!'
        }
        failure {
            echo 'Pipeline failed — check console output above.'
        }
        always {
            sh 'docker logout $ACR_LOGIN_SERVER || true'
        }
    }
}