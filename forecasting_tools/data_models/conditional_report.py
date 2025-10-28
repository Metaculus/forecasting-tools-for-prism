from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.data_models.conditional_models import (
    ConditionalPrediction,
    ConditionalPredictionTypes,
)
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import (
    ConditionalQuestion,
    MetaculusQuestion,
)


class ConditionalReport(ForecastReport):
    question: ConditionalQuestion
    prediction: ConditionalPrediction

    @staticmethod
    async def _get_aggregates_for_question(
        question: MetaculusQuestion, forecasts: list[ConditionalPredictionTypes]
    ):
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        parent_report_type = DataOrganizer.get_report_type_for_question_type(
            type(question)
        )
        return await parent_report_type.aggregate_predictions(forecasts, question)

    @classmethod
    async def aggregate_predictions(
        cls, predictions: list[ConditionalPrediction], question: ConditionalQuestion
    ) -> ConditionalPrediction:

        parent_forecasts = [prediction.parent for prediction in predictions]
        aggregated_parent = await cls._get_aggregates_for_question(
            question.parent, parent_forecasts
        )

        child_forecasts = [prediction.child for prediction in predictions]
        aggregated_child = await cls._get_aggregates_for_question(
            question.child, child_forecasts
        )

        yes_forecasts = [prediction.prediction_yes for prediction in predictions]
        aggregated_yes = await cls._get_aggregates_for_question(
            question.question_yes, yes_forecasts
        )

        no_forecasts = [prediction.prediction_no for prediction in predictions]
        aggregated_no = await cls._get_aggregates_for_question(
            question.question_no, no_forecasts
        )

        return ConditionalPrediction(
            parent=aggregated_parent,
            child=aggregated_child,
            prediction_yes=aggregated_yes,  # type: ignore
            prediction_no=aggregated_no,  # type: ignore
        )

    @classmethod
    def make_readable_prediction(cls, prediction: ConditionalPrediction) -> str:
        return clean_indents(
            f"""
            Parent forecast: {prediction.parent}
            Child forecast: {prediction.child}
            Yes forecast: {prediction.prediction_yes}
            No forecast: {prediction.prediction_no}
        """
        )

    async def publish_report_to_metaculus(self) -> None:
        # TODO: implement
        raise NotImplementedError()
