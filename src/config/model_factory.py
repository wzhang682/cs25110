from typing import Dict, Any, Union
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
# from langchain_anthropic import ChatAnthropic
from . import config

class ModelFactory:
    @staticmethod
    def create_model(model_config: Union[Dict[str, Any], str]) -> Union[ChatOpenAI,BaseChatOpenAI]:#Union[ChatOpenAI, ChatAnthropic]
        """
        Create LLM model based on provider configuration.
        
        Args:
            model_config: Either a dict with provider/model/temperature or a string model name (legacy)
            
        Returns:
            Configured LLM instance
        """
        # Handle legacy string model names
        if isinstance(model_config, str):
            return BaseChatOpenAI(
                    model='deepseek-reasoner', 
                    openai_api_key='sk-203c68a7572b4c27a62c93b1b17417f5', 
                    openai_api_base='https://api.deepseek.com',
                    temperature=0.7
                )
        
        # Handle new dict-based configuration
        provider = model_config.get('provider')
        model_name = model_config.get('model')
        temperature = model_config.get('temperature')
        
        if not model_name:
            raise ValueError("Model name is required in configuration")
        if provider == 'deepseek':
            return BaseChatOpenAI(
                model='deepseek-chat', 
                openai_api_key='sk-203c68a7572b4c27a62c93b1b17417f5', 
                openai_api_base='https://api.deepseek.com',
                temperature=temperature
            )
        elif provider == 'deepseek-chat':     
            return BaseChatOpenAI(
                model='deepseek-chat', 
                openai_api_key='sk-203c68a7572b4c27a62c93b1b17417f5', 
                openai_api_base='https://api.deepseek.com',
                temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: openai, anthropic,deepseek")
    
    @staticmethod
    def get_portfolio_manager_model():
        """Get configured portfolio manager model."""
        return ModelFactory.create_model(config.model_portfolio_manager)
    
    @staticmethod
    def get_nlp_features_model():
        """Get configured NLP features model."""
        return ModelFactory.create_model(config.model_nlp_features)
    
    @staticmethod
    def get_assess_significance_model():
        """Get configured assess significance model."""
        return ModelFactory.create_model(config.model_assess_significance)
    
    @staticmethod
    def get_enhanced_summary_model():
        """Get configured enhanced summary model."""
        return ModelFactory.create_model(config.model_enhanced_summary) 