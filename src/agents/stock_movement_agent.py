import os
from typing import Dict, Any, List, Optional
from ..workflows.state import AgentState
from ..tools.finnhub_tool import get_company_news
from ..tools.firecrawl_tool import scrape_url
from ..config.model_factory import ModelFactory
from ..config import config

from .news_intelligence_agent import (
    sample_random_news,
    assess_significance,
    scrape_article_content,
    create_enhanced_summary,
)

from ..prompts import (
    get_news_analysis_template,
    get_news_output_parser,
    format_news_data,
    format_significant_news_data,
    clean_input_string,
    get_stock_movement_template,
)

async def analyze_stockmovement(
    symbol: str, 
    analysis_date: Optional[str] = None, 
    technical_data: Optional[Dict[str, Any]] = None,
    company_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Complete news analysis workflow:
    1. Get news -> 2. Sample random articles -> 3. Assess significance 
    4. Scrape significant articles -> 5. Generate NLP features
    """
    try:
        symbol = symbol.upper()
        
        # Extract company information from data_collection_results
        company_info = None
        if company_data:
            company_info = company_data.get('company_info') or company_data.get('company_profile')
        
        # 1. Get news data for analysis
        news_result = await get_company_news(symbol, analysis_date)
        
        # Extract news data
        news_data = []
        if news_result and news_result.success and news_result.data:
            if isinstance(news_result.data, dict):
                news_data = news_result.data.get('news', [])
            elif isinstance(news_result.data, list):
                news_data = news_result.data
        
        if not news_data:
            return {
                'symbol': symbol,
                'success': False,
                'error': 'No news data available'
            }
        
        # 2. Sample random news items per config
        sampled_news = sample_random_news(news_data, max_count=config.news_sample_count)
        
        # 3. Process each news item for significance with company context
        processed_news = []
        significant_news = []
        
        for news_item in sampled_news:
            # Assess significance with company context
            significance = await assess_significance(
                news_item.get('headline', ''),
                news_item.get('summary', ''),
                news_item.get('date', ''),
                symbol,
                company_info
            )
            
            news_item['significance_score'] = significance
            
            # 4. If significant (threshold per config), scrape and enhance
            if significance >= config.news_significance_threshold:
                full_content = await scrape_article_content(news_item)
                if full_content:
                    enhanced_summary = await create_enhanced_summary(
                        news_item.get('headline', ''),
                        news_item.get('summary', ''),
                        full_content,
                        symbol
                    )
                    if enhanced_summary:
                        news_item['enhanced_summary'] = enhanced_summary
                        significant_news.append(news_item)
            
            processed_news.append(news_item)
        
        # 5. Generate NLP features
        nlp_features = await extract_nlp_features(
            symbol, 
            processed_news, 
            significant_news,
            technical_data
        )
        
         # 6. Predict stock movement
        movement=await predict_stock_movement(
            symbol, 
            processed_news, 
            significant_news,
            technical_data
        )

        if not nlp_features:
            return {
                'symbol': symbol,
                'success': False,
                'error': 'NLP feature extraction failed'
            }
        
        return {
            'symbol': symbol,
            'nlp_features': nlp_features,
            'news_count': len(processed_news),
            'significant_count': len(significant_news),
            'movement':movement,
            'success': True
        }
        
    except Exception as e:
        print(f"Error in news analysis for {symbol}: {e}")
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }

async def predict_stock_movement(
    symbol: str,
    all_news: List[Dict],
    significant_news: List[Dict],
    technical_data: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, int]]:
    """Predict stock movement direction"""
    try:
        api_key = config.get_api_key('openai')
        if not api_key:
            print(f"No OpenAI API key available for {symbol}")
            return None

        os.environ['OPENAI_API_KEY'] = api_key
        llm = ModelFactory.get_nlp_features_model()

        prompt_template = get_stock_movement_template()
        output_parser = get_news_output_parser()

        moderate_news = [
            item for item in all_news
            if item not in significant_news
            and item.get('significance_score', 0.0) >= config.news_moderate_threshold
        ]

        formatted_regular_news = clean_input_string(format_news_data(moderate_news))
        formatted_significant_news = clean_input_string(format_significant_news_data(significant_news))

        prompt_input = {
            'symbol': symbol,
            'news': formatted_regular_news,
            'significant_news': formatted_significant_news,
            'format_instructions': output_parser.get_format_instructions()
        }

        chain = prompt_template | llm | output_parser
        result = await chain.ainvoke(prompt_input)

        # 假设 result 是一个 dict，里面至少包含 'direction' 字段
        if not isinstance(result, dict):
            print(f"Invalid result format: {type(result)}")
            return None

        if result not in ('rise', 'fall'):
            print(f"Invalid direction value: {result}")
            return None

        return result
    
    except Exception as e:
        print(f"Error predicting stock movement for {symbol}: {e}")
        return None

async def extract_nlp_features(
    symbol: str,
    all_news: List[Dict],
    significant_news: List[Dict],
    technical_data: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, int]]:
    """Extract 7 NLP features using proper separation of regular vs significant news."""
    try:
        api_key = config.get_api_key('openai')
        
        if not api_key:
            print(f"No OpenAI API key available for {symbol}")
            return None
        
        os.environ['OPENAI_API_KEY'] = api_key
        llm = ModelFactory.get_nlp_features_model()
        
        # Prepare prompts and data
        prompt_template = get_news_analysis_template()
        output_parser = get_news_output_parser()
        
        # Separate regular and significant news, filtering out low significance items
        # Only include news with significance >= moderate_threshold in the analysis
        moderate_news = [item for item in all_news if item not in significant_news and item.get('significance_score', 0.0) >= config.news_moderate_threshold]
        
        # Format data (company_info removed - focusing only on news content)
        formatted_regular_news = format_news_data(moderate_news)
        formatted_significant_news = format_significant_news_data(significant_news)
        
        # Clean strings
        formatted_regular_news = clean_input_string(formatted_regular_news)
        formatted_significant_news = clean_input_string(formatted_significant_news)
        
        # Prepare prompt input
        prompt_input = {
            'symbol': symbol,
            'news': formatted_regular_news,
            'significant_news': formatted_significant_news,
            'format_instructions': output_parser.get_format_instructions()
        }
        
        # Execute chain
        chain = prompt_template | llm | output_parser
        result = await chain.ainvoke(prompt_input)
        
        # Validate and convert results
        if isinstance(result, dict):
            nlp_features = {}
            required_features = [
                'news_relevance', 'sentiment', 'price_impact_potential',
                'trend_direction', 'earnings_impact', 'investor_confidence', 
                'risk_profile_change'
            ]
            
            for feature in required_features:
                value = result.get(feature)
                if value is not None:
                    try:
                        int_value = int(value)
                        # Validate that value is in allowed range: -2, -1, 0, 1, 2
                        if int_value not in [-2, -1, 0, 1, 2]:
                            print(f"Invalid {feature} value: {int_value} (must be -2, -1, 0, 1, or 2)")
                            return None
                        nlp_features[feature] = int_value
                    except (ValueError, TypeError):
                        print(f"Invalid {feature} value: {value} (must be integer)")
                        return None
                else:
                    print(f"Missing {feature} in result")
                    return None
            
            return nlp_features
        else:
            print(f"Invalid result format: {type(result)}")
            return None
        
    except Exception as e:
        print(f"Error extracting NLP features for {symbol}: {e}")
        return None


async def stock_movement_agent_node(state: AgentState) -> AgentState:
    """
    LangGraph node for news intelligence with complete workflow.
    """
    try:
        # Get symbol, analysis_date and technical data from previous agents
        symbol = state['symbols'][0] if state['symbols'] else 'AAPL'
        analysis_date = state['analysis_date']
        technical_data = state.get('technical_analysis_results')
        
        # Get company data from data collection results
        company_data = state.get('data_collection_results')
        
        # Perform complete news analysis with company context
        result = await analyze_stockmovement(symbol, analysis_date, technical_data, company_data)
        
        # Update state
        state['stock_movement_results'] = result
        state['current_step'] = 'stock_movement_complete'
        
        if not result['success']:
            state['error'] = result.get('error', 'stock movement failed')
            
        return state
        
    except Exception as e:
        print(f"Stock movement node error: {e}")
        state['error'] = str(e)
        state['current_step'] = 'error'
        return state 