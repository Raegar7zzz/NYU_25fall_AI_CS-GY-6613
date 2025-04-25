(define (problem route_two_requests)
  (:domain openrouter_routing)

  (:objects
    ; Providers
    openai cohere local - provider

    ; LLMs
    gpt4_openai textdavinci openai - llm
    cohere_lamp cohere_xlarge - llm
    local_vicuna - llm

    ; Accounts
    acct_openai acct_cohere acct_local - account

    ; Capabilities
    code multilingual safe_kids long_context - capability

    ; Requests
    code_req translate_req - request
  )

  (:init
    ; Providers → LLMs
    (llm_of_provider gpt4_openai openai)
    (llm_of_provider textdavinci openai)
    (llm_of_provider cohere_lamp cohere)
    (llm_of_provider cohere_xlarge cohere)
    (llm_of_provider local_vicuna local)

    ; Accounts → Providers
    (account_of_provider acct_openai openai)
    (account_of_provider acct_cohere cohere)
    (account_of_provider acct_local local)

    ; LLM Capabilities
    (has_capability gpt4_openai code)
    (has_capability gpt4_openai long_context)
    (has_capability textdavinci code)
    (has_capability cohere_lamp multilingual)
    (has_capability cohere_xlarge multilingual)
    (has_capability local_vicuna code)
    (has_capability local_vicuna safe_kids)

    ; Requests → Required Capabilities
    (requires_capability code_req code)
    (requires_capability translate_req multilingual)

    ; can_handle = requires_capability ∧ token‐limit checks (simplified as same)
    (can_handle gpt4_openai code_req)
    (can_handle textdavinci code_req)
    (can_handle cohere_lamp translate_req)
    (can_handle cohere_xlarge translate_req)
    (can_handle local_vicuna code_req)

    ; Costs ($/million tokens)
    (= (cost_per_mil gpt4_openai) 6.00)
    (= (cost_per_mil textdavinci) 0.02)
    (= (cost_per_mil cohere_lamp) 0.75)
    (= (cost_per_mil cohere_xlarge) 1.00)
    (= (cost_per_mil local_vicuna) 0.00)

    ; Account balances
    (= (balance acct_openai) 10.0)
    (= (balance acct_cohere) 5.0)
    (= (balance acct_local) 1.0)

    ; cost_ok if balance > cost for 1k tokens = 0.001 * cost_per_mil
    (cost_ok acct_openai gpt4_openai)
    (cost_ok acct_openai textdavinci)
    (cost_ok acct_cohere cohere_lamp)
    (cost_ok acct_cohere cohere_xlarge)
    (cost_ok acct_local local_vicuna)
  )

  (:goal (and
    (assigned code_req ?l1 ?a1)
    (assigned translate_req ?l2 ?a2)
  ))
)