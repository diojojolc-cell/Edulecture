from config.base_config import Config



class ModelFactory:
    @staticmethod
    def get_model(config: Config):
        if config.arch == 'baseline_8_squential_itm':
            from mutil_grade.model.baseline_8_sequential_itm import Baseline_8_sequential
            return Baseline_8_sequential(config)
        elif config.arch == 'baseline_8_squential_MG':
            from model.baseline_8_sequential_MG import Baseline_8_sequential
            return Baseline_8_sequential(config)
        elif config.arch == 'baseline_8_squential_ch':
            from model.baseline_8_sequential_ch import Baseline_8_sequential
            return Baseline_8_sequential(config)
        elif config.arch == 'baseline_8_squential' :
            from model.baseline_8_sequential import Baseline_8_sequential
            return Baseline_8_sequential(config)
        elif config.arch == 'baseline_8_squential_itm_ch':
            from model.baseline_8_sequential_itm_ch import Baseline_8_sequential
            return Baseline_8_sequential(config)
        elif config.arch == 'baseline_8_evidence_transformer_itm':
            from model.baseline_8_evidence_transformer_itm import Baseline_8_evidence_transformer_itm
            return Baseline_8_evidence_transformer_itm(config)
        elif config.arch == 'baseline_8_clip_only':
            from model.baseline_8_clip_only import Baseline_8_clip_only
            return Baseline_8_clip_only(config)
        else:
            raise NotImplementedError
