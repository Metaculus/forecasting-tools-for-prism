from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.data_models.conditional_models import ConditionalPrediction
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.questions import ConditionalQuestion


class ConditionalReport(ForecastReport):
    question: ConditionalQuestion
    prediction: ConditionalPrediction

    @classmethod
    async def aggregate_predictions(
        cls, predictions: list[ConditionalPrediction], question: ConditionalQuestion
    ) -> ConditionalPrediction:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        parent_forecasts = [prediction.parent for prediction in predictions]
        parent_report_type = DataOrganizer.get_report_type_for_question_type(
            type(question.parent)
        )
        aggregated_parent = await parent_report_type.aggregate_predictions(
            parent_forecasts, question.parent
        )

        child_forecasts = [prediction.child for prediction in predictions]
        child_report_type = DataOrganizer.get_report_type_for_question_type(
            type(question.child)
        )
        aggregated_child = await child_report_type.aggregate_predictions(
            child_forecasts, question.child
        )

        yes_forecasts = [prediction.prediction_yes for prediction in predictions]
        yes_report_type = DataOrganizer.get_report_type_for_question_type(
            type(question.question_yes)
        )
        aggregated_yes = await yes_report_type.aggregate_predictions(
            yes_forecasts, question.question_yes
        )

        no_forecasts = [prediction.prediction_no for prediction in predictions]
        no_report_type = DataOrganizer.get_report_type_for_question_type(
            type(question.question_no)
        )
        aggregated_no = await no_report_type.aggregate_predictions(
            no_forecasts, question.question_no
        )

        return ConditionalPrediction(
            parent=aggregated_parent,
            child=aggregated_child,
            prediction_yes=aggregated_yes,
            prediction_no=aggregated_no,
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
