from sklearn import pipeline, pipeline
from sklearn.base import clone
from sklearn.utils import _print_elapsed_time
from sklearn.utils.validation import check_memory
from sklearn.pipeline import _fit_transform_one, _routing_enabled
from sklearn.utils.metadata_routing import process_routing

class Pipeline(pipeline.Pipeline):

    def _fit(self, X, y=None, routed_params=None):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time("Pipeline", self._log_message(step_idx)):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                # we do not clone when caching is disabled to
                # preserve backward compatibility
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            # Fit or load from cache the current transformer
            X, fitted_transformer = fit_transform_one_cached(
                cloned_transformer,
                X,
                y,
                None,
                message_clsname="Pipeline",
                message=self._log_message(step_idx),
                params=routed_params[name],
            )


            if isinstance(X, tuple):
                X, y = X

            # print(X.shape, y.shape)

            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        return X, y
    
    def fit(self, X, y=None, **params):
        routed_params = self._check_method_params(method="fit", props=params)
        Xt = self._fit(X, y, routed_params)

        if isinstance(Xt, tuple):    ###### unpack X if is tuple: X = (X,y)
            Xt, y = Xt

        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                last_step_params = routed_params[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **last_step_params["fit"])

        return self

    def score(self, X, y=None, sample_weight=None, **params):
        Xt = X
        if not _routing_enabled():
            for _, name, transform in self._iter(with_final=False):
                Xt = transform.transform(Xt)
                if isinstance(Xt, tuple):    ###### unpack X if is tuple: X = (X,y)
                    Xt, y = Xt
            score_params = {}
            if sample_weight is not None:
                score_params["sample_weight"] = sample_weight
            return self.steps[-1][1].score(Xt, y, **score_params)

        # metadata routing is enabled.
        routed_params = process_routing(
            self, "score", sample_weight=sample_weight, **params
        )

        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt, **routed_params[name].transform)
        return self.steps[-1][1].score(Xt, y, **routed_params[self.steps[-1][0]].score)