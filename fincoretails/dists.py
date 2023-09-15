"""
Little container module that wraps all the distribution modules.
"""
import fincoretails.unipareto
import fincoretails.powpareto
import fincoretails.algpareto
import fincoretails.expareto
import fincoretails.general_algpareto
import fincoretails.general_powpareto
import fincoretails.general_expareto

import fincoretails.lognormal
import fincoretails.santafe

distributions = [
             fincoretails.unipareto,
             fincoretails.algpareto,
             fincoretails.powpareto,
             fincoretails.expareto,
             fincoretails.general_algpareto,
             fincoretails.general_powpareto,
             fincoretails.general_expareto,
             fincoretails.lognormal,
             fincoretails.santafe,
        ]

