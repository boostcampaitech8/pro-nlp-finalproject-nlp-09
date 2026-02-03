## Set up ADC for a local development environment
Google Cloud SDK(gcloud)를 사용하여 로컬 개발 환경에서 Application Default Credentials(ADC)를 설정하는 방법은 다음과 같습니다.
### 1. gcloud 설치
- [gcloud 설치 가이드](https://docs.cloud.google.com/sdk/docs/install-sdk)를 참고하여 gcloud를 설치합니다.

### 2. gcloud 초기화
- 터미널(명령 프롬프트)에서 다음 명령어를 입력하여 gcloud를 초기화합니다.
```bash
gcloud init
```
- 명령어를 입력하면 브라우저가 열리며 Google 계정으로 로그인하라는 메시지가 표시됩니다. 로그인 후 터미널로 돌아가서 프로젝트를 선택하거나 새 프로젝트를 만듭니다.

### 3. gcloud 인증
- 다음 명령어를 입력하여 gcloud 인증을 수행합니다.  
```bash
gcloud auth application-default login
```
- 이 명령어를 입력하면 다시 브라우저가 열리며 Google 계정으로 로그인하라는 메시지가 표시됩니다. 로그인 후 터미널로 돌아가면 인증이 완료됩니다.
- 가이드: [gcloud 인증 가이드](https://docs.cloud.google.com/compute/docs/gcloud-compute)

### 4. ADC 토큰 파기
- 로컬 자격증명이 필요하지 않다면 다음 명령어를 입력하여 ADC 토큰을 파기할 수 있습니다.
```bash
gcloud auth application-default revoke
```