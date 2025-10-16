from fastapi import HTTPException, status

def handle_openai_errors(error):
        if error.status_code == 429:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"OpenAI Rate Limit Error: {error.message}"
            )
        elif error.status_code == 401:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"OpenAI Authentication Error: {error.message}"
            )
        elif error.status_code == 400:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"OpenAI Invalid Request Error: {error.message}"
            )
        elif error.status_code  == 503:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"OpenAI Service Unavailable Error: {error.message}"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"OpenAI API Error: {error.message}"
            )
