(define (domain openrouter_routing)
  (:requirements :typing :fluents)
  (:types llm provider account capability request)

  (:predicates
    (llm_of_provider ?l - llm ?p - provider)
    (has_capability ?l - llm ?c - capability)
    (account_of_provider ?a - account ?p - provider)
    (requires_capability ?r - request ?c - capability)
    (can_handle ?l - llm ?r - request)
    (cost_ok ?a - account ?l - llm)
    (assigned ?r - request ?l - llm ?a - account)
  )

  (:functions
    (cost_per_mil ?l - llm) ; $/million tokens
    (balance ?a - account)  ; $ remaining
  )

  (:action route_request
    :parameters (?r - request ?l - llm ?a - account ?p - provider)
    :precondition (and
      (llm_of_provider ?l ?p)
      (account_of_provider ?a ?p)
      (can_handle ?l ?r)
      (cost_ok ?a ?l)
    )
    :effect (and
      (assigned ?r ?l ?a)
      (decrease (balance ?a) (* 0.001 (cost_per_mil ?l)))
    )
  )
)