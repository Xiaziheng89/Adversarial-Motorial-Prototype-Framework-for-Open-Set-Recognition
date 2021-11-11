
def consistency_detection(**options):
    if options['cs'] and options['cs++']:
        raise Exception("The '--cs' and '--cs++' cannot exist simultaneously!")

    elif options['cs']:
        if options['loss'] != 'ARPLoss' and options['loss'] != 'AMPFLoss':
            raise Exception('The loss function does not match the experimental Settings!')

    elif options['cs++']:
        if options['loss'] != 'AMPFLoss':
            raise Exception('The loss function does not match the experimental Settings!')
